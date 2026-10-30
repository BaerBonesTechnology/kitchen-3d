# SPDX-License-Identifier: GPL-2.0-or-later
"""Binary persistence and native Cycles integration regression, isolated via runner."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import statistics
import sys
import time

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/modules'))
sys.path.insert(0, str(Path(__file__).parent))
import kitchen3d_lighting as lighting
from kitchen3d_lighting import rend, cycles_engine
from kitchen3d_persistent_lighting import make_scene, state, aim


def rejected(function):
    try:
        function()
    except (ValueError, OSError):
        return
    raise AssertionError('Invalid cache was accepted')


def assert_equal(a, b):
    assert a.patches.keys() == b.patches.keys()
    for key, entry in a.entries.items():
        other = b.entries[key]
        for name in ('direct', 'radiance'):
            assert np.array_equal(entry[name], other[name]), name
        for name in ('links', 'segments', 'transport_segments'):
            assert entry[name] == other[name], name


def configure_materials(scene):
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        for slot in obj.material_slots:
            mat = slot.material
            mat.use_nodes = True
            mat.node_tree.nodes.clear()
            diffuse = mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
            diffuse.inputs['Color'].default_value = mat.diffuse_color
            output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
            mat.node_tree.links.new(diffuse.outputs[0], output.inputs['Surface'])
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = 0
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.samples = 16
    scene.cycles.device = 'CPU'


def child():
    if not hasattr(bpy.types.Scene, 'kitchen3d_lighting'):
        lighting.register()
    scene = bpy.context.scene
    assert scene.render.engine == cycles_engine.ENGINE
    cache = rend.load(cycles_engine.cache_path(scene))
    cache.sync(*state())
    assert not cache.dirty and cache.rays['lighting'] == 0
    cycles_engine.check_prepared(scene, bpy.context.evaluated_depsgraph_get())
    print('PASS: .REND reused after fresh process reload')


def main():
    output = Path(os.environ.get('KITCHEN3D_TEST_OUTPUT', '/tmp/kitchen3d-rend'))
    output.mkdir(parents=True, exist_ok=True)
    lighting.register()
    scene, light, blocker = make_scene()
    configure_materials(scene)
    scene.render.resolution_x = scene.render.resolution_y = int(os.environ.get('KITCHEN3D_REND_BENCH_SIZE', '64'))
    scene.cycles.samples = int(os.environ.get('KITCHEN3D_REND_BENCH_SAMPLES', '16'))
    scene.kitchen3d_lighting.cache_path = str(output / 'lighting.REND')
    cache = lighting.Cache(64)
    cache.sync(*state())
    rejected(lambda: rend.save(output / 'incomplete.REND', cache))
    cache.update(100000)
    started = time.perf_counter()
    rend.save(output / 'lighting.REND', cache)
    write_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    restored = rend.load(output / 'lighting.REND')
    read_ms = (time.perf_counter() - started) * 1000
    assert_equal(cache, restored)
    assert restored.rays['lighting'] == 0
    restored.sync(*state())
    assert not restored.dirty
    for angle in (0, 1, 2):
        aim(scene, angle)
        restored.sync(*state())
        assert not restored.dirty
        assert np.array_equal(cache.render(scene, 32, 32)[0], restored.render(scene, 32, 32)[0])
    blob = (output / 'lighting.REND').read_bytes()
    for name, data in [('truncated', blob[:50]), ('payload_truncated', blob[:-7]),
                       ('checksum', blob[:-1] + bytes([blob[-1] ^ 1])), ('trailing', blob + b'x'),
                       ('version', blob[:8] + (999).to_bytes(4, 'little') + blob[12:])]:
        path = output / (name + '.REND')
        path.write_bytes(data)
        rejected(lambda: rend.load(path))
    rejected(lambda: rend.save(output / 'wrong.blend', cache))
    # A valid checksum does not make malformed dependency lengths acceptable.
    magic, version, msize, size, digest = rend.HEADER.unpack_from(blob)
    body = bytearray(blob[rend.HEADER.size:])
    body[msize + 18 * 8:msize + 18 * 8 + 4] = (2**32 - 1).to_bytes(4, 'little')
    path = output / 'bad_index.REND'
    path.write_bytes(rend.HEADER.pack(magic, version, msize, size, hashlib.sha256(body).digest()) + body)
    rejected(lambda: rend.load(path))
    # Selective invalidation still operates after disk restore.
    light.location.x += 1
    restored.sync(*state())
    assert restored.dirty
    restored.update(100000)
    fresh = lighting.Cache(64)
    fresh.sync(*state()); fresh.update(100000)
    assert_equal(fresh, restored)
    light.location.x -= 1
    bpy.context.view_layer.update()
    cycles_engine.prepare(scene, bpy.context.evaluated_depsgraph_get(), load_only=True)
    bpy.context.view_layer.update()
    cycles_engine.check_prepared(scene, bpy.context.evaluated_depsgraph_get())
    aim(scene, 0)
    scene.render.engine = cycles_engine.ENGINE
    scene.kitchen3d_lighting.cache_path = '//lighting.REND'
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'persistent-cycles.blend'))
    # A separate process ensures session UIDs cannot accidentally provide persistence.
    result = subprocess.run([bpy.app.binary_path, '--background', '--factory-startup',
                             '--python-expr',
                             f'import sys; sys.path.insert(0, {str(ROOT / "scripts/modules")!r}); import kitchen3d_lighting; kitchen3d_lighting.register()',
                             str(output / 'persistent-cycles.blend'), '--python-exit-code', '1',
                             '--python', str(Path(__file__).resolve()), '--', '--reload-check'],
                            capture_output=True, text=True, timeout=60)
    (output / 'reload.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and 'PASS:' in result.stdout, result.stdout + result.stderr
    import _cycles
    native = getattr(_cycles, 'kitchen3d_persistent_cache_version', 0) == 1
    timings = {}
    if native:
        for engine, name in [('CYCLES', 'cycles'), (cycles_engine.ENGINE, 'persistent_cycles')]:
            scene.render.engine = engine
            scene.render.filepath = str(output / (name + '.exr'))
            scene.render.image_settings.file_format = 'OPEN_EXR'
            start = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            timings[name] = (time.perf_counter() - start) * 1000
        a = bpy.data.images.load(str(output / 'cycles.exr'))
        b = bpy.data.images.load(str(output / 'persistent_cycles.exr'))
        va, vb = np.asarray(a.pixels[:]), np.asarray(b.pixels[:])
        assert va.size and vb.size and np.max(vb) > 0
        rgb_a, rgb_b = va.reshape(-1, 4)[:, :3], vb.reshape(-1, 4)[:, :3]
        timings['rmse_linear_rgb_vs_cycles'] = float(np.sqrt(np.mean((rgb_a - rgb_b)**2)))
        timings['relative_rmse_vs_cycles'] = timings['rmse_linear_rgb_vs_cycles'] / max(1e-8, float(np.sqrt(np.mean(rgb_a**2))))
        # Alternating modes avoids attributing all system warm-up to one engine.
        samples = {'cycles': [], 'persistent_cycles': []}
        for repeat in range(4):
            aim(scene, .1 * (repeat + 1))
            order = [('CYCLES', 'cycles'), (cycles_engine.ENGINE, 'persistent_cycles')]
            if repeat % 2:
                order.reverse()
            for engine, name in order:
                scene.render.engine = engine
                start = time.perf_counter()
                bpy.ops.render.render()
                samples[name].append((time.perf_counter() - start) * 1000)
        timings['warm_camera_rerenders_ms'] = samples
        timings['warm_medians_ms'] = {k: statistics.median(v) for k, v in samples.items()}
        aim(scene, 0)
        scene.render.engine = cycles_engine.ENGINE

        # Prove the native lookup changes indirect transport, rather than just engine labeling.
        # This uses the native Cycles base engine with the new ID but bypasses wrapper validation
        # only in this test to inject a zero-cache sentinel.
        original_check = cycles_engine.check_prepared
        try:
            cycles_engine.check_prepared = lambda *args: None
            for obj in scene.objects:
                if obj.type == 'MESH':
                    attr = obj.data.attributes[cycles_engine.ATTRIBUTE]
                    attr.data.foreach_set('vector', np.zeros(len(attr.data) * 3, dtype=np.float32))
                    obj.data.update()
            bpy.context.view_layer.update()
            scene.render.filepath = str(output / 'zero_cache.exr')
            bpy.ops.render.render(write_still=True)
            z = bpy.data.images.load(str(output / 'zero_cache.exr'))
            vz = np.asarray(z.pixels[:])
            assert np.max(np.abs(vz - vb)) > 1e-5, 'Native cache hook had no effect'
        finally:
            cycles_engine.check_prepared = original_check
        cycles_engine.prepare(scene, bpy.context.evaluated_depsgraph_get(), load_only=True)
    blocker.location.x += .5
    bpy.context.view_layer.update()
    rejected(lambda: cycles_engine.check_prepared(scene, bpy.context.evaluated_depsgraph_get()))
    report = {'resolution': scene.render.resolution_x, 'cycles_samples': scene.cycles.samples,
              'file_bytes': len(blob), 'patches': len(cache.patches), 'write_ms': write_ms,
              'read_ms_including_bvh': read_ms, 'reload_lighting_rays': 0, 'native_tested': native,
              'native_render': timings, 'binary': bpy.app.binary_path,
              'build_hash': bpy.app.build_hash.decode()}
    (output / 'rend-metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: .REND roundtrip, fresh-process persistence, corruption rejection, invalidation;', report)


if '--reload-check' in sys.argv:
    child()
else:
    main()
