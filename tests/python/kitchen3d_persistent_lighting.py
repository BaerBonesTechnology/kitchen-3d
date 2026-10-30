# SPDX-License-Identifier: GPL-2.0-or-later
"""Run in Kitchen 3D: --background --factory-startup --python THIS -- --output DIR."""
import argparse
import json
import math
import os
import platform
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/modules'))
import kitchen3d_lighting as lighting


def make_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    def cube(name, location, scale, color, cuts):
        bpy.ops.mesh.primitive_cube_add(size=2, location=location)
        obj = bpy.context.object
        obj.name = name
        obj.scale = scale
        material = bpy.data.materials.new(name)
        material.diffuse_color = (*color, 1)
        obj.data.materials.append(material)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=cuts)
        bpy.ops.object.mode_set(mode='OBJECT')
        return obj
    cube('Floor', (0, 0, -.15), (3, 3, .15), (.65, .65, .65), 5)
    cube('Red', (-1.1, 0, .65), (.55, .65, .65), (.75, .05, .04), 2)
    cube('Blue', (1.1, .3, .8), (.6, .55, .8), (.04, .12, .75), 2)
    blocker = cube('Green blocker', (0, 1.5, .8), (.35, .45, .8), (.08, .65, .12), 2)
    bpy.ops.object.light_add(type='POINT', location=(-1, -1, 4))
    light = bpy.context.object
    light.data.energy = 700
    light.data.shadow_soft_size = 0
    bpy.ops.object.camera_add(location=(5, -7, 5))
    scene.camera = bpy.context.object
    scene.camera.data.lens = 42
    aim(scene, 0)
    scene.render.resolution_x = scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Standard'
    scene.world.color = (0, 0, 0)
    return scene, light, blocker


def aim(scene, angle):
    camera = scene.camera
    camera.location = (7 * math.sin(angle), -7 * math.cos(angle), 4.5)
    camera.rotation_euler = (Vector((0, 0, .6)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()


def state():
    bpy.context.view_layer.update()
    return lighting.snapshot(bpy.context.evaluated_depsgraph_get())


def save_image(path, pixels):
    h, w, _ = pixels.shape
    image = bpy.data.images.new(path.stem, w, h, float_buffer=True)
    image.pixels.foreach_set(pixels.ravel())
    image.filepath_raw = str(path)
    image.file_format = 'PNG'
    image.save()
    bpy.data.images.remove(image)


def error(a, b):
    diff = a[:, :, :3].astype(np.float64) - b[:, :, :3]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {'rmse_linear_rgb': rmse,
            'relative_rmse': rmse / max(1e-8, float(np.sqrt(np.mean(b[:, :, :3] ** 2)))),
            'max_abs_linear_rgb': float(np.max(np.abs(diff)))}


def frame(cache, scene, rebuild=False, freeze=False):
    start = time.perf_counter()
    if rebuild:
        cache = lighting.Cache(cache.samples)
    before = cache.rays.copy()
    cache.sync(*state())
    sync_ms = (time.perf_counter() - start) * 1000
    begin = time.perf_counter()
    cache.update(100000, freeze)
    update_ms = (time.perf_counter() - begin) * 1000
    begin = time.perf_counter()
    pixels, missing = cache.render(scene, 64, 64)
    render_ms = (time.perf_counter() - begin) * 1000
    return cache, pixels, dict(frame_ms=(time.perf_counter() - start) * 1000,
                               sync_ms=sync_ms, update_ms=update_ms, camera_ms=render_ms,
                               lighting_rays=cache.rays['lighting'] - before['lighting'],
                               camera_rays=cache.rays['camera'] - before['camera'],
                               invalidated=cache.last_invalidated, pending=len(cache.dirty), missing_pixels=missing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('KITCHEN3D_TEST_OUTPUT', '/tmp/kitchen3d-persistent-lighting')))
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    args.output.mkdir(parents=True, exist_ok=True)
    scene, light, blocker = make_scene()
    lighting.register()
    cache, initial, warm = frame(lighting.Cache(64), scene)
    assert not cache.dirty
    frozen_values = {k: v['radiance'].copy() for k, v in cache.entries.items()}
    frames = []
    for i in range(8):
        aim(scene, 2 * math.pi * i / 8)
        cache, persistent, reuse = frame(cache, scene, freeze=True)
        _, rebuilt, baseline = frame(cache, scene, rebuild=True)
        assert reuse['lighting_rays'] == 0 and reuse['invalidated'] == 0
        assert reuse['missing_pixels'] == 0
        assert np.array_equal(persistent, rebuilt)
        assert all(np.array_equal(v, cache.entries[k]['radiance']) for k, v in frozen_values.items())
        frames.append({'orbit_index': i, 'reuse': reuse, 'rebuild': baseline,
                       'reuse_vs_rebuild': error(persistent, rebuilt)})
        save_image(args.output / f'orbit_{i:02}.png', persistent)
    aim(scene, 0)
    _, initial, _ = frame(cache, scene, freeze=True)
    # Independent per-pixel one-bounce path estimator: actual hit points, no patch-cache reads.
    start = time.perf_counter()
    reference, _ = cache.render(scene, 64, 64, reference_samples=256)
    reference_ms = (time.perf_counter() - start) * 1000
    save_image(args.output / 'reference.png', reference)
    save_image(args.output / 'cached.png', initial)
    indirect, _ = cache.render(scene, 64, 64, indirect_only=True)
    indirect_reference, _ = cache.render(scene, 64, 64, reference_samples=1024, indirect_only=True)
    save_image(args.output / 'indirect.png', indirect)
    save_image(args.output / 'indirect_reference.png', indirect_reference)
    reference_rays = cache.rays['reference']

    def subdivide_blocker():
        bpy.ops.object.select_all(action='DESELECT')
        blocker.select_set(True)
        bpy.context.view_layer.objects.active = blocker
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=1)
        bpy.ops.object.mode_set(mode='OBJECT')

    mutations = []
    for name, mutate in [
        ('light_move', lambda: setattr(light, 'location', (2, -1, 3))),
        ('blocker_into_paths', lambda: setattr(blocker, 'location', (0, -.6, 1.5))),
        ('blocker_out_of_paths', lambda: setattr(blocker, 'location', (0, 1.5, .8))),
        ('material_change', lambda: setattr(blocker.data.materials[0], 'diffuse_color', (.7, .15, .02, 1))),
        ('deformation', lambda: blocker.data.vertices[0].co.__setitem__(0, -.8)),
        ('topology_change', subdivide_blocker),
        ('object_add', lambda: bpy.ops.mesh.primitive_cube_add(size=.6, location=(0, 0, 1))),
        ('object_delete', lambda: bpy.data.objects.remove(blocker, do_unlink=True)),
    ]:
        old_links = {k: tuple(e['links']) for k, e in cache.entries.items()}
        mutate()
        begin = time.perf_counter()
        cache.sync(*state())
        invalidation_ms = (time.perf_counter() - begin) * 1000
        pending = len(cache.dirty)
        before = cache.rays['lighting']
        assert cache.update(2) <= 2
        assert cache.rays['lighting'] - before <= 2 * (cache.samples + len(cache.lights))
        assert len(cache.dirty) >= pending - 2
        cache, pixels, stats = frame(cache, scene)
        fresh, expected, _ = frame(cache, scene, rebuild=True)
        difference = error(pixels, expected)
        changed_connections = sum(old_links.get(k) != tuple(e['links']) for k, e in cache.entries.items())
        total_update_rays = cache.rays['lighting'] - before
        if name == 'light_move':
            assert changed_connections == 0
            assert total_update_rays <= len(cache.patches) * len(cache.lights)
        if name in {'blocker_into_paths', 'blocker_out_of_paths'}:
            assert changed_connections > 0
        dynamic_reference, _ = cache.render(scene, 64, 64, reference_samples=128)

        # Compare ALL patches, including currently invisible receivers.
        max_patch_error = max(float(np.max(np.abs(entry['radiance'] - fresh.entries[k]['radiance'])))
                              for k, entry in cache.entries.items())
        assert max_patch_error < 1e-6, (name, max_patch_error)
        assert difference['max_abs_linear_rgb'] < 1e-6, (name, difference)
        mutations.append({'name': name, 'initial_dirty_patches': pending, 'invalidation_ms': invalidation_ms, 'total_patches': len(cache.patches),
                          'max_patch_error_vs_rebuild': max_patch_error,
                          'total_lighting_update_rays': total_update_rays, 'changed_connection_lists': changed_connections,
                          'image_error_vs_reference_128spp': error(pixels, dynamic_reference), 'image_error_vs_rebuild': difference,
                          'completion_frame': stats})
        save_image(args.output / f'{name}.png', pixels)
    # Missing coverage, freeze behavior, lifecycle and actual RenderEngine output.
    empty = lighting.Cache(64)
    empty.sync(*state())
    before = empty.rays.copy()
    assert empty.update(freeze=True) == 0 and before == empty.rays
    _, missing = empty.render(scene, 16, 16)
    assert missing > 0
    scene.render.engine = 'KITCHEN3D_PERSISTENT'
    bpy.ops.render.render()
    assert lighting._caches
    engine_cache = next(iter(lighting._caches.values()))
    assert engine_cache.entries and not engine_cache.dirty
    lighting_rays = engine_cache.rays['lighting']
    scene.kitchen3d_lighting.freeze = True
    aim(scene, .2)
    bpy.ops.render.render()
    assert next(iter(lighting._caches.values())) is engine_cache
    assert engine_cache.rays['lighting'] == lighting_rays
    lighting.clear_caches()
    assert not lighting._caches
    # Separately measure Python allocation peak so instrumentation does not distort frame timing.
    tracemalloc.start()
    measured = lighting.Cache(64)
    measured.sync(*state())
    measured.update(100000)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {'environment': {'blender': bpy.app.version_string, 'binary': bpy.app.binary_path,
                               'build_hash': bpy.app.build_hash.decode(), 'platform': platform.platform(),
                               'processor': platform.processor(), 'device': 'CPU / Python / mathutils BVH'},
              'configuration': {'width': 64, 'height': 64, 'connections_per_patch': 64,
                                'reference_samples_per_pixel': 256, 'diffuse_indirect_bounces': 1},
              'warmup': warm, 'orbit_frames': frames, 'mutations': mutations,
              'reference': {'time_ms': reference_ms, 'combined_and_indirect_reference_rays': reference_rays,
                            'indirect_cached_vs_reference_1024spp': error(indirect, indirect_reference),
                            'cached_vs_reference': error(initial, reference),
                            'description': 'Independent uncached one-bounce CPU path estimator; finite-sample reference'},
              'memory_after_mutations': {'patches': len(measured.patches),
                                        'logical_packed_payload_bytes': measured.payload_bytes(),
                                        'python_traced_current_bytes': current, 'python_traced_peak_bytes': peak,
                                        'excludes': 'BVH/native allocations, interpreter, images; not GPU VRAM'},
              'summary': {'reuse_median_ms': statistics.median(f['reuse']['frame_ms'] for f in frames),
                          'rebuild_median_ms': statistics.median(f['rebuild']['frame_ms'] for f in frames)}}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    # Restore the full scene for a usable example .blend, with the experimental engine selected.
    scene, _, _ = make_scene()
    scene.render.engine = 'KITCHEN3D_PERSISTENT'
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / 'persistent-lighting.blend'))
    lighting.unregister()
    print('PASS: persistent lighting:', json.dumps(report['summary']), 'output:', args.output)


if __name__ == '__main__':
    main()
