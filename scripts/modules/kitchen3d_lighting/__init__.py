# SPDX-License-Identifier: GPL-2.0-or-later
"""Opt-in CPU experiment. Import and call register(); never enabled by startup."""
import hashlib
import json
import math
import time
from dataclasses import dataclass

import bpy
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Vector
from mathutils.bvhtree import BVHTree

EPS = 1e-4
FAR = 1000.0


@dataclass
class Patch:
    key: tuple
    vertices: tuple
    color: tuple

    @property
    def center(self):
        return sum((Vector(v) for v in self.vertices), Vector()) / 3

    @property
    def normal(self):
        a, b, c = map(Vector, self.vertices)
        return (b - a).cross(c - a).normalized()


def object_key(obj):
    original = obj.original
    return json.dumps([original.library.filepath if original.library else '', original.name_full],
                      separators=(',', ':'))


def snapshot(depsgraph):
    """Camera-independent evaluated triangles. No shader interpretation or instances."""
    patches, lights = {}, []
    for instance in depsgraph.object_instances:
        if instance.is_instance:
            raise ValueError('Persistent Lighting does not support instances; realize them first')
        obj = instance.object
        if obj.hide_render:
            continue
        if obj.type == 'LIGHT':
            if obj.data.type != 'POINT':
                raise ValueError('Persistent Lighting supports point lights only')
            lights.append((tuple(obj.matrix_world.translation),
                           tuple(c * obj.data.energy / (4 * math.pi) for c in obj.data.color)))
        elif obj.type == 'MESH':
            mesh = obj.to_mesh()
            try:
                mesh.calc_loop_triangles()
                topology = hashlib.sha256(np.array(
                    [tuple(t.vertices) for t in mesh.loop_triangles], dtype='<u4').tobytes()).digest()
                # Content identity invalidates triangle indices after topology edits.
                for i, tri in enumerate(mesh.loop_triangles):
                    material = obj.material_slots[tri.material_index].material if obj.material_slots else None
                    color = tuple(material.diffuse_color[:3]) if material else (0.7,) * 3
                    key = (object_key(obj), topology, i)
                    vertices = tuple(tuple(obj.matrix_world @ mesh.vertices[j].co) for j in tri.vertices)
                    patch = Patch(key, vertices, color)
                    if patch.normal.length > 0:
                        patches[key] = patch
            finally:
                obj.to_mesh_clear()
        elif obj.type not in {'CAMERA', 'EMPTY', 'ARMATURE'}:
            raise ValueError('Persistent Lighting supports mesh geometry only')
    return patches, tuple(sorted(lights))


def directions(normal, count, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((count, 2))
    tangent = normal.cross(Vector((0, 0, 1)) if abs(normal.z) < 0.9 else Vector((0, 1, 0))).normalized()
    bitangent = normal.cross(tangent)
    for u, v in xy:
        r, phi = math.sqrt(u), 2 * math.pi * v
        yield (r * math.cos(phi) * tangent + r * math.sin(phi) * bitangent + math.sqrt(1 - u) * normal)


def overlaps_segment(start, end, lo, hi):
    """Conservative segment/AABB slab test, including misses and shadow rays."""
    tmin, tmax = 0.0, 1.0
    for a, b, lower, upper in zip(start, end, lo, hi):
        d = b - a
        if abs(d) < 1e-12:
            if a < lower or a > upper:
                return False
        else:
            t0, t1 = (lower - a) / d, (upper - a) / d
            tmin, tmax = max(tmin, min(t0, t1)), min(tmax, max(t0, t1))
            if tmin > tmax:
                return False
    return True


class Cache:
    def __init__(self, samples=64):
        self.samples = samples
        self.patches = {}
        self.lights = ()
        self.entries = {}
        self.dirty = set()
        self.transport_dirty = set()
        self.bvh = None
        self.keys = []
        self.rays = {'lighting': 0, 'camera': 0, 'reference': 0}
        self.last_invalidated = 0

    def sync(self, patches, lights):
        geometry = {k for k in self.patches.keys() | patches.keys()
                    if k not in self.patches or k not in patches or
                    self.patches[k].vertices != patches[k].vertices}
        changed = geometry | {k for k in self.patches.keys() & patches.keys()
                              if self.patches[k].color != patches[k].color}
        light_changed = lights != self.lights
        dirty = geometry & patches.keys()
        if geometry:
            # Group changed triangles by object; old+new bounds cover vacated AND occupied space.
            boxes = {}
            for collection in (self.patches, patches):
                for key in geometry & collection.keys():
                    boxes.setdefault(key[0], []).extend(collection[key].vertices)
            bounds = [(np.min(v, axis=0) - EPS, np.max(v, axis=0) + EPS) for v in boxes.values()]
            for key, entry in self.entries.items():
                if any(overlaps_segment(a, b, lo, hi)
                       for a, b in entry['segments'] for lo, hi in bounds):
                    dirty.add(key)
            self.keys = list(patches)
            verts = [v for p in patches.values() for v in p.vertices]
            self.bvh = BVHTree.FromPolygons(verts, [(i, i + 1, i + 2) for i in range(0, len(verts), 3)],
                                            all_triangles=True) if verts else None
        transport_dirty = dirty.copy()
        dirty.update(changed & patches.keys())
        if light_changed:
            dirty.update(patches)
        for key in self.entries.keys() - patches.keys():
            del self.entries[key]
        self.patches, self.lights = patches, lights
        self.dirty = (self.dirty | dirty) & patches.keys()
        self.transport_dirty = (self.transport_dirty | transport_dirty) & patches.keys()
        self.last_invalidated = len(dirty & patches.keys())

    def cast(self, start, direction, distance=FAR, category='lighting'):
        self.rays[category] += 1
        if self.bvh is None:
            return None, None
        loc, normal, index, dist = self.bvh.ray_cast(start, direction, distance)
        return (self.keys[index], loc) if index is not None else (None, None)

    def direct(self, point, normal, color, category='lighting', segments=None):
        value = np.zeros(3)
        start = point + normal * EPS
        for position, intensity in self.lights:
            delta = Vector(position) - start
            distance = delta.length
            if distance < EPS:
                continue
            direction = delta / distance
            cosine = max(0.0, normal.dot(direction))
            if cosine == 0:
                continue
            # Retain the complete shadow segment even when blocked.
            if segments is not None:
                segments.append((tuple(start), tuple(position)))
            hit, _ = self.cast(start, direction, distance - EPS, category)
            if hit is None:
                value += np.array(color) * np.array(intensity) * cosine / (math.pi * distance * distance)
        return value

    def update(self, budget=256, freeze=False, cancelled=lambda: False):
        if freeze or budget < 1:
            return 0
        count = 0
        for key in self.keys:
            if cancelled():
                break
            if key not in self.dirty:
                continue
            patch = self.patches[key]
            point, normal = patch.center, patch.normal
            segments, links = [], []
            direct = self.direct(point, normal, patch.color, segments=segments)
            if key in self.transport_dirty or key not in self.entries:
                transport_segments = []
                # Same quadrature on rebuild, independent of camera and scheduling.
                for direction in directions(normal, self.samples, key[2] + 1701):
                    start = point + normal * EPS
                    hit, location = self.cast(start, direction)
                    transport_segments.append((tuple(start), tuple(
                        location if hit is not None else start + direction * FAR)))
                    if hit is not None and self.patches[hit].normal.dot(-direction) > 0:
                        links.append(hit)
            else:
                links = self.entries[key]['links']
                transport_segments = self.entries[key]['transport_segments']
            segments.extend(transport_segments)
            self.entries[key] = {'direct': direct, 'links': links, 'segments': segments,
                                 'transport_segments': transport_segments, 'radiance': direct.copy()}
            self.dirty.remove(key)
            self.transport_dirty.discard(key)
            count += 1
            if count >= budget:
                break
        if count:
            # One bounce: dependencies read DIRECT radiance, never recursively feed themselves.
            # Global O(N*K) resolve is conservative; no all-pairs graph or extra rays.
            for key, entry in self.entries.items():
                incoming = sum((self.entries[hit]['direct'] for hit in entry['links']
                                if hit in self.entries), np.zeros(3)) / self.samples
                entry['radiance'] = entry['direct'] + np.array(self.patches[key].color) * incoming
        return count

    def render(self, scene, width, height, reference_samples=0, indirect_only=False, cancelled=lambda: False):
        camera = scene.camera
        if camera is None or camera.data.type != 'PERSP':
            raise ValueError('A perspective camera is required')
        frame = camera.data.view_frame(scene=scene)
        left, right = min(v.x / -v.z for v in frame), max(v.x / -v.z for v in frame)
        bottom, top = min(v.y / -v.z for v in frame), max(v.y / -v.z for v in frame)
        rotation = camera.matrix_world.to_quaternion()
        origin = camera.matrix_world.translation
        pixels = np.zeros((height, width, 4), dtype=np.float32)
        pixels[:, :, 3] = 1
        missing = 0
        for y in range(height):
            if cancelled():
                break
            for x in range(width):
                direction = rotation @ Vector((left + (right - left) * (x + .5) / width,
                                                bottom + (top - bottom) * (y + .5) / height, -1)).normalized()
                key, point = self.cast(origin, direction, category='camera')
                if key is None:
                    continue
                patch = self.patches[key]
                if patch.normal.dot(-direction) <= 0:
                    continue
                if reference_samples:
                    value = (np.zeros(3) if indirect_only else
                             self.direct(point, patch.normal, patch.color, 'reference'))
                    incoming = np.zeros(3)
                    for ray in directions(patch.normal, reference_samples, 9173 + y * width + x):
                        hit, pos = self.cast(point + patch.normal * EPS, ray, category='reference')
                        if hit is not None and self.patches[hit].normal.dot(-ray) > 0:
                            target = self.patches[hit]
                            incoming += self.direct(pos, target.normal, target.color, 'reference')
                    value += np.array(patch.color) * incoming / reference_samples
                elif key in self.entries:
                    value = self.entries[key]['radiance']
                    if indirect_only:
                        value = value - self.entries[key]['direct']
                else:
                    missing += 1
                    value = (1, 0, 1)  # Explicit missing coverage, not an invented ambient term.
                pixels[y, x, :3] = value
        return pixels, missing

    def payload_bytes(self):
        """Logical packed payload estimate, excludes Python, BVH and allocator overhead."""
        return len(self.patches) * (9 * 4 + 3 * 4 + 8) + sum(
            6 * 4 + len(e['links']) * 8 + len(e['segments']) * 6 * 4 for e in self.entries.values())


_caches = {}


@persistent
def clear_caches(*_):
    _caches.clear()
    from . import cycles_engine
    cycles_engine.clear_loaded()


class PersistentLightingEngine(bpy.types.RenderEngine):
    bl_idname = 'KITCHEN3D_PERSISTENT'
    bl_label = 'Kitchen 3D Persistent Lighting (CPU Experiment)'
    bl_use_preview = False

    def render(self, depsgraph):
        scene = depsgraph.scene_eval
        settings = scene.kitchen3d_lighting
        key = scene.original.session_uid
        cache = _caches.get(key)
        if cache is None or cache.samples != settings.samples or settings.rebuild:
            cache = _caches[key] = Cache(settings.samples)
        started = time.perf_counter()
        try:
            cache.sync(*snapshot(depsgraph))
            cache.update(settings.budget, settings.freeze, self.test_break)
            scale = scene.render.resolution_percentage / 100
            width, height = int(scene.render.resolution_x * scale), int(scene.render.resolution_y * scale)
            pixels, missing = cache.render(scene, width, height, cancelled=self.test_break)
            if self.test_break():
                return
            result = self.begin_result(0, 0, width, height)
            result.layers[0].passes['Combined'].rect = pixels.reshape(-1, 4)
            self.end_result(result)
            self.update_stats('Persistent Lighting',
                              f'{time.perf_counter() - started:.3f}s | {len(cache.dirty)} pending | {missing} missing pixels')
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))


class LightingSettings(bpy.types.PropertyGroup):
    cache_path: bpy.props.StringProperty(
        name="Lighting cache", subtype="FILE_PATH", default="//lighting.REND",
        options=({'PATH_SUPPORTS_BLEND_RELATIVE'}
                 if 'is_path_supports_blend_relative' in bpy.types.Property.bl_rna.properties else set()))
    samples: bpy.props.IntProperty(name='Connections per patch', default=64, min=1, max=4096)
    budget: bpy.props.IntProperty(name='Patch updates per render', default=4096, min=1, max=100000)
    freeze: bpy.props.BoolProperty(name='Freeze lighting updates', default=False)
    rebuild: bpy.props.BoolProperty(name='Rebuild every render (baseline)', default=False)


class KITCHEN3D_PT_lighting(bpy.types.Panel):
    bl_label = 'Persistent Lighting Experiment'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'KITCHEN3D_PERSISTENT'

    def draw(self, context):
        for prop in ('samples', 'budget', 'freeze', 'rebuild'):
            self.layout.prop(context.scene.kitchen3d_lighting, prop)
        self.layout.label(text='CPU / one diffuse bounce / point lights')
        self.layout.label(text='Uses Material viewport color; ignores shader nodes')
        self.layout.label(text='F12 renders; no live rendered viewport')


_CLASSES = (LightingSettings, PersistentLightingEngine, KITCHEN3D_PT_lighting)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.kitchen3d_lighting = bpy.props.PointerProperty(type=LightingSettings)
    from . import cycles_engine
    cycles_engine.register()
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        handlers.append(clear_caches)


def unregister():
    from . import cycles_engine
    cycles_engine.unregister()
    clear_caches()
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if clear_caches in handlers:
            handlers.remove(clear_caches)
    del bpy.types.Scene.kitchen3d_lighting
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
