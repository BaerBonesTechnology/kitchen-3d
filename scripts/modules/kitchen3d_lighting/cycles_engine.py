# SPDX-License-Identifier: GPL-2.0-or-later
"""Separate Cycles engine using the native diffuse-hit cache hook."""
from pathlib import Path

import bpy
import cycles
import numpy as np

from . import Cache, object_key, snapshot
from . import rend

ENGINE = 'KITCHEN3D_CYCLES_PERSISTENT'
ATTRIBUTE = '.kitchen3d_radiance'
_loaded = {}


def clear_loaded():
    _loaded.clear()


def cache_path(scene):
    path = scene.kitchen3d_lighting.cache_path
    if not path:
        raise ValueError('Choose a .REND cache path in Render Properties')
    if path.startswith('//') and not bpy.data.filepath:
        raise ValueError('Save the .blend project before using a relative cache path')
    return Path(bpy.path.abspath(path))


def validate_scene(scene, depsgraph):
    """Fail closed: the initial cache estimator is not a general Cycles shader evaluator."""
    if scene.cycles.device != 'CPU':
        raise ValueError('Persistent Cycles is CPU-only until GPU kernels and packaging are validated')
    if scene.render.use_motion_blur or scene.cycles.shading_system:
        raise ValueError('Persistent Cycles currently requires motion blur and OSL to be disabled')
    if scene.cycles.use_guiding:
        raise ValueError('Disable path guiding for this experimental cache')
    if scene.world and scene.world.use_nodes:
        nodes = scene.world.node_tree.nodes
        outputs = [n for n in nodes if n.type == 'OUTPUT_WORLD' and n.is_active_output]
        if not outputs or outputs[0].inputs['Volume'].is_linked:
            raise ValueError('Persistent Cycles requires a black, non-volume world')
        surface = outputs[0].inputs['Surface']
        if surface.is_linked:
            node = surface.links[0].from_node
            if (node.type != 'BACKGROUND' or node.inputs['Color'].is_linked or
                    node.inputs['Strength'].is_linked or
                    (node.inputs['Strength'].default_value != 0 and
                     any(node.inputs['Color'].default_value[:3]))):
                raise ValueError('Persistent Cycles requires a black world')
    elif scene.world and any(scene.world.color):
        raise ValueError('Persistent Cycles requires a black world')
    meshes = set()
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.hide_render:
            continue
        if obj.type == 'LIGHT' and obj.data.shadow_soft_size != 0:
            raise ValueError('Use point lights with Radius = 0')
        if obj.type != 'MESH':
            continue
        original = obj.original
        if original.data.as_pointer() in meshes:
            raise ValueError('Make mesh data single-user before preparing the cache')
        meshes.add(original.data.as_pointer())
        if original.library or original.data.library or original.modifiers:
            raise ValueError('Use local meshes with applied modifiers for this experiment')
        if any(p.use_smooth for p in original.data.polygons):
            raise ValueError('Use flat shading for this experiment')
        if not original.material_slots:
            raise ValueError('Assign a plain Diffuse BSDF material to each mesh')
        for slot in original.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                raise ValueError('Materials must use a plain Diffuse BSDF')
            outputs = [n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
            if not outputs:
                raise ValueError('Material needs an active output')
            out = outputs[0]
            if out.inputs['Volume'].is_linked or out.inputs['Displacement'].is_linked:
                raise ValueError('Volume and displacement shaders are unsupported by the cache')
            links = out.inputs['Surface'].links
            if not links or links[0].from_node.type != 'BSDF_DIFFUSE':
                raise ValueError('Only a plain Diffuse BSDF is supported by the cache builder')
            node = links[0].from_node
            if any(s.is_linked for s in node.inputs) or node.inputs['Roughness'].default_value != 0:
                raise ValueError('Diffuse BSDF must have constant color, zero roughness and no normal input')
            if not np.allclose(node.inputs['Color'].default_value[:3], mat.diffuse_color[:3], atol=1e-7):
                raise ValueError('Match Material viewport color to the Diffuse BSDF color')


def face_values(obj, cache):
    mesh = obj.data
    mesh.calc_loop_triangles()
    values = np.zeros((len(mesh.polygons), 3), dtype=np.float64)
    counts = np.zeros(len(mesh.polygons), dtype=np.int32)
    name = object_key(obj)
    entries = {k[2]: entry for k, entry in cache.entries.items() if k[0] == name}
    for i, tri in enumerate(mesh.loop_triangles):
        if i not in entries:
            raise ValueError('Degenerate/missing triangle: rebuild valid geometry before caching')
        values[tri.polygon_index] += entries[i]['radiance']
        counts[tri.polygon_index] += 1
    return (values / np.maximum(1, counts[:, None])).astype(np.float32)


def prepare(scene, depsgraph, load_only=False):
    validate_scene(scene, depsgraph)
    path = cache_path(scene)
    patches, lights = snapshot(depsgraph)
    if path.exists():
        cache = rend.load(path)
        if cache.samples != scene.kitchen3d_lighting.samples:
            if load_only:
                raise ValueError('Cache sample count differs from the scene setting')
            cache = Cache(scene.kitchen3d_lighting.samples)
    elif load_only:
        raise ValueError('The .REND file does not exist')
    else:
        cache = Cache(scene.kitchen3d_lighting.samples)
    cache.sync(patches, lights)
    if load_only and cache.dirty:
        raise ValueError('The .REND cache does not match the current scene; use Build / Update')
    if not load_only:
        cache.update(len(patches) + 1)
    # Compute every output before editing any mesh attributes.
    names = {k[0] for k in patches}
    outputs = [(obj, face_values(obj, cache)) for obj in scene.objects
               if obj.type == 'MESH' and object_key(obj) in names]
    for obj, values in outputs:
        attribute = obj.data.attributes.get(ATTRIBUTE)
        if attribute and (attribute.domain != 'FACE' or attribute.data_type != 'FLOAT_VECTOR'):
            raise ValueError('Reserved .kitchen3d_radiance attribute has an incompatible type')
    if not load_only:
        rend.save(path, cache)
    clear_loaded()
    for obj, values in outputs:
        attribute = obj.data.attributes.get(ATTRIBUTE) or obj.data.attributes.new(ATTRIBUTE, 'FLOAT_VECTOR', 'FACE')
        attribute.data.foreach_set('vector', values.ravel())
        obj.data.update()
    return cache


def check_prepared(scene, depsgraph):
    validate_scene(scene, depsgraph)
    path = cache_path(scene)
    stat = path.stat()
    stamp = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if stamp not in _loaded:
        clear_loaded()
        _loaded[stamp] = rend.load(path)
    cache = _loaded[stamp]
    if cache.samples != scene.kitchen3d_lighting.samples:
        raise ValueError('Sample count changed; Build / Update the .REND cache')
    cache.sync(*snapshot(depsgraph))
    if cache.dirty:
        clear_loaded()
        raise ValueError('Scene lighting changed; Build / Update the .REND cache before rendering')
    names = {k[0] for k in cache.patches}
    for obj in scene.objects:
        if obj.type != 'MESH' or object_key(obj) not in names:
            continue
        attribute = obj.data.attributes.get(ATTRIBUTE)
        if attribute is None or attribute.domain != 'FACE' or attribute.data_type != 'FLOAT_VECTOR':
            raise ValueError('Load / Apply the .REND cache before rendering')
        values = np.empty((len(obj.data.polygons), 3), dtype=np.float32)
        attribute.data.foreach_get('vector', values.ravel())
        if not np.array_equal(values, face_values(obj, cache)):
            raise ValueError('Mesh cache attributes differ from .REND; Load / Apply the cache')
    return cache


class KitchenPersistentCycles(bpy.types.RenderEngine):
    bl_idname = ENGINE
    bl_label = 'Kitchen 3D Persistent Cycles (Experimental)'
    bl_use_preview = False
    bl_use_eevee_viewport = True
    bl_use_exclude_layers = True
    bl_use_spherical_stereo = True
    bl_use_custom_freestyle = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = None
        self.cache_ready = False

    def __del__(self):
        cycles.engine.free(self)

    # Reuse Cycles' integration callbacks without subclassing its registered RNA
    # class (that replaces the CYCLES RNA Python class and breaks the original engine).
    render_frame_finish = cycles.CyclesRender.render_frame_finish
    update_render_passes = cycles.CyclesRender.update_render_passes
    draw = cycles.CyclesRender.draw

    def update(self, data, depsgraph):
        import _cycles
        self.cache_ready = False
        try:
            if getattr(_cycles, 'kitchen3d_persistent_cache_version', 0) != 1:
                raise ValueError('This engine requires the rebuilt Kitchen 3D native cache kernel')
            check_prepared(depsgraph.scene, depsgraph)
        except (ValueError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return
        self.cache_ready = True
        cycles.CyclesRender.update(self, data, depsgraph)

    def render(self, depsgraph):
        if self.cache_ready:
            cycles.engine.render(self, depsgraph)

    def view_update(self, context, depsgraph):
        # Explicitly disabled until bounded background preparation is implemented.
        self.report({'ERROR'}, 'Use F12 for Persistent Cycles; live viewport caching is not implemented')

    def view_draw(self, context, depsgraph):
        pass


class KITCHEN3D_OT_prepare_rend(bpy.types.Operator):
    bl_idname = 'kitchen3d.prepare_rend'
    bl_label = 'Build / Update .REND'
    bl_options = {'REGISTER'}
    load_only: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        try:
            cache = prepare(context.scene, context.evaluated_depsgraph_get(), self.load_only)
        except (ValueError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f'{len(cache.entries)} patches; {cache.rays["lighting"]} lighting rays')
        return {'FINISHED'}


class KITCHEN3D_PT_rend(bpy.types.Panel):
    bl_label = 'Persistent Cycles / .REND Cache'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == ENGINE

    def draw(self, context):
        settings = context.scene.kitchen3d_lighting
        self.layout.prop(settings, 'cache_path')
        self.layout.prop(settings, 'samples')
        self.layout.operator('kitchen3d.prepare_rend')
        self.layout.operator('kitchen3d.prepare_rend', text='Load / Apply .REND').load_only = True
        self.layout.prop(context.scene.cycles, 'samples')
        self.layout.prop(context.scene.cycles, 'device')
        self.layout.label(text='Flat diffuse meshes, black world, point lights')
        self.layout.label(text='Prepare after edits, then F12. Native rebuild required.')


CLASSES = (KitchenPersistentCycles, KITCHEN3D_OT_prepare_rend, KITCHEN3D_PT_rend)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
