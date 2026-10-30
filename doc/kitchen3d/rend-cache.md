# Persistent Cycles and binary .REND caches

Kitchen 3D now has a separate experimental engine ID,
`KITCHEN3D_CYCLES_PERSISTENT`, labelled **Kitchen 3D Persistent Cycles
(Experimental)**. It reuses Cycles session callbacks and uses a native Cycles kernel
hook, while registering independently from Cycles' RNA class. Ordinary `CYCLES` does not enable the hook. This is a prototype within the
Kitchen 3D source tree, not a separate complete fork of every Cycles subsystem.

A normal `.blend` remains the project file. The binary `.REND` sidecar stores the
lighting representation, not a rendered image. `Scene.kitchen3d_lighting.cache_path`
saves its reference in the project. Mesh attributes are an upload copy and do not
replace the authoritative sidecar. Copy both files when moving the project.
Relative `//lighting.REND` paths work once the project has been saved.

## Use

This engine requires a Kitchen 3D binary built with the new native kernel. The
previous installed application can test cache IO but cannot render with this hook.
The Python wrapper checks `_cycles.kitchen3d_persistent_cache_version` and reports
an error if the binary is incompatible.

The tested developer app is:
`/Users/jamiewill/Documents/Projects/build_kitchen3d_5.3/bin/Kitchen 3D.app`.
It was built in place; runtime resources were copied from the existing installed
app and the new Python module added. The installed `/Applications` app was not
replaced. This is not a signed release package. The two pre-existing deleted plist
templates were restored only during compilation and returned to their deleted state.

For the generated sample, launch the rebuilt app with registration **before** opening
its project (the new module is included in that app's Python resources):

```sh
'/Users/jamiewill/Documents/Projects/build_kitchen3d_5.3/bin/Kitchen 3D.app/Contents/MacOS/Kitchen 3D' \
  --python-expr 'import kitchen3d_lighting; kitchen3d_lighting.register()' \
  /Users/jamiewill/Documents/Projects/build_kitchen3d_5.3/tests/rend-cache-native/persistent-cycles.blend
```

Alternatively, register the opt-in module once in the Python Console before opening
a saved project (the path is for this checkout):

```python
import sys
sys.path.insert(0, '/Users/jamiewill/Documents/Projects/blaya/scripts/modules')
import kitchen3d_lighting
kitchen3d_lighting.register()
```

1. Save the project normally as `.blend`.
2. Choose **Kitchen 3D Persistent Cycles (Experimental)** as the render engine.
3. In Render Properties choose a `.REND` path and connections per patch.
4. Click **Build / Update .REND**. This completes the cache and uploads it to mesh attributes.
5. Press F12. Camera moves need no lighting update.
6. After geometry, materials or lights change, click **Build / Update .REND** again.
7. After reopening, register the module and use **Load / Apply .REND** if needed.
   A saved project normally already contains the upload attributes, but the engine
   still checks them against the sidecar before rendering.

Build/Update loads an existing sidecar and selectively repairs dependencies when
possible. Changing the sample count rebuilds the representation. Load/Apply never
silently rebuilds an incompatible scene. Corrupt files are reported rather than
overwritten; select a new path if rebuilding from a corrupt sidecar is intended.
Neither operation changes source material graphs or triangulates the source mesh.

Registration remains session-only and is not automatically installed into startup.
The original CPU prototype remains available as a separate engine. No live rendered
viewport is implemented for Persistent Cycles yet. Use F12.

## Native transport integration

The standard Cycles session, BVH traversal, camera sampling and film are retained.
At a front-facing triangle hit reached by a diffuse reflection, the new kernel
looks for `ATTR_STD_KITCHEN3D_RADIANCE`. A valid nonnegative finite value replaces
the remaining path and is multiplied by the existing path throughput. It must not
be multiplied by the hit BSDF or light MIS a second time. Camera, glossy,
transmission and subsurface hits do not take this branch; missing native attributes
fall back to ordinary Cycles tracing.

The Python wrapper intentionally rejects missing/stale cache uploads before render,
so native fallback is defensive, not a promise of automatic mixed-scene support.
Light groups, light/shadow linking, custom visibility and specialized render passes
have not been validated; use ordinary visible diffuse meshes and the Combined pass.
Path guiding, OSL, motion blur and GPU device selection are rejected in this first integration.
Only CPU kernels have been exercised; GPU rendering must not silently use old packaged kernels.

The cache builder is still the earlier CPU centroid/point-light estimator. It stores
direct plus one indirect diffuse bounce. Querying that cache after a Cycles bounce
therefore approximates additional diffuse transport; it is not equivalent to either
the old one-bounce image renderer or unrestricted converged Cycles. The builder is
not yet using Cycles shader evaluation or GPU ray tracing.

`.REND` retains triangle values. For native upload, triangles belonging to the same
Blender polygon are averaged into a reserved FACE/FLOAT_VECTOR attribute named
`.kitchen3d_radiance`. Cycles expands the value onto its triangles. This extra spatial
approximation is explicit; triangulating/subdividing the input before caching can
reduce it. Future integration should use per-triangle buffers rather than polygon
averages. No claim about RX580 support or performance follows from this change.

## Supported preparation scene

Use local, single-user, flat-shaded meshes with applied modifiers, a black world,
and point lights with zero radius. Every material must have a constant-color,
zero-roughness Diffuse BSDF connected directly to the active output, with no linked
normal, volume or displacement. Match its viewport color to the BSDF color because
the CPU builder reads viewport color. Unsupported shader configurations are rejected.

Instances and non-mesh render geometry remain unsupported. Original mesh names and
library identifiers are part of persistent identity; renaming conservatively
invalidates those surfaces. Topology connectivity has a SHA-256 fingerprint.
World-space positions, albedo and lights are compared on load/update. Camera
transforms do not participate. Surface IDs survive application restart without
relying on Blender's transient session UIDs.

## Binary format v1

All numeric fields are little-endian. No pickle or executable objects are loaded.
The current prototype caps the complete metadata/payload at 256 MiB, metadata at
8 MiB, surface count at 100,000, and samples per patch at 4,096.

| Header field | Encoding |
| --- | --- |
| Magic | 8 bytes, `K3DREND` followed by NUL |
| Version | uint32, currently 1 |
| Metadata length | uint32 |
| Payload length | uint64 |
| Body checksum | 32-byte SHA-256 of metadata plus payload |

A bounded UTF-8 JSON metadata section records estimator revision, epsilon, ray
extent, sample count, lights and stable surface keys. Lighting/dependency values
are binary. Each surface record contains:

- 18 float64 values: triangle positions (9), albedo (3), direct radiance (3),
  resolved radiance (3).
- Three uint32 counts: hit indices, total segments, transport segments.
- uint32 hit indices into the surface table, preserving duplicate sampling weights.
- float64 segment endpoints, six numbers per segment. Transport segments are the
  final K entries; shadow segments precede them. Misses are retained.

Float64 preserves the CPU prototype's values exactly across save/load. This is not
a compressed, quantized or GPU-native format. GPU upload converts face averages to
float32. The checksum detects accidental corruption; it is not an authenticity
signature. Version or estimator mismatches, invalid indices/counts, nonfinite values,
truncation, trailing data and duplicate keys are rejected before use.

Writes use a temporary file in the destination directory, flush/fsync, and atomic
replacement. A failed write leaves the previous complete cache in place. The parent
directory must exist. Only `.REND` extensions (case-insensitive) are accepted; a
cache is never written over a `.blend` file.

On loading, the CPU BVH is reconstructed without tracing lighting rays. In-memory
cache data is retained between F12 calls while path, size and nanosecond modification
time are unchanged; load/undo/redo clears that copy. Scene and upload validation
still run before rendering. This maintenance and file-loading time is not free.

## Tests and reproducibility

```sh
python3 tests/python/run_kitchen3d_test.py \
  --blender '/path/to/rebuilt/Kitchen 3D.app/Contents/MacOS/Kitchen 3D' \
  --script tests/python/kitchen3d_rend_cache.py \
  --output /Users/jamiewill/Documents/Projects/build_kitchen3d_5.3/tests/rend-cache-native
```

The test covers exact binary round-trip, fresh-process `.blend`/`.REND` reload,
frozen camera views, malformed files, selective updates after reload, stale-scene
rejection and native render output. On a compatible rebuilt binary it renders with
ordinary Cycles and Persistent Cycles, then injects a zero cache in the isolated
test scene to verify that the native hook actually changes transport. It saves
EXR comparisons and JSON measurements. On an older binary, IO tests run and the
report explicitly records `native_tested: false`; that is not native validation.
Both this suite and the original persistence benchmark are registered under the
`kitchen3d` CTest label. Their isolated runner has a 180-second timeout.

Timing a small first render includes session initialization and kernel startup.
It is not a GPU-throughput benchmark. Use repeated warmed renders and target-device
validation before drawing performance conclusions.


## Measured result, 2026-10-29

The native build and integration test passed, including ordinary Cycles rendering,
Persistent Cycles rendering, a zero-cache transport-effect check and persistence
across a fresh process. The earlier CPU persistence suite also passed after changing
surface identity. Full upstream tests, GPU kernels and interactive viewport use were
not validated.

| CPU workload | Ordinary Cycles warmed median | Persistent Cycles warmed median |
| --- | ---: | ---: |
| 64×64, 16 samples | 11.6 ms | 19.9 ms |
| 256×256, 128 samples | 625.4 ms | 553.8 ms |

Four orbit rerenders per mode were measured, alternating order, with both modes
using the same scene/sample settings and Blender's default Persistent Data disabled.
These are single-run developer measurements, not confidence-bounded benchmarks.
The small case is slower because validation overhead dominates. In the larger case,
median wall time fell by about 11.4%. The 256px linear-RGB relative RMSE against the
128-sample Cycles image was 1.95%; this is an approximation, not equal-quality proof.
The Cycles image itself is a finite-sample reference.

The 756-patch `.REND` occupies 2,546,381 bytes. On the larger test run, saving took
49.4 ms and loading including CPU BVH reconstruction took 127.1 ms, with zero
lighting rays. Those costs are separate from warmed rerenders. The first render
also includes disk loading unless the cache has already been loaded during validation.
Repeated rendering amortizes cache creation/load; the file alone does not guarantee
less total time. A larger scene, adaptive patch density and comparison with Cycles'
built-in Persistent Data enabled should be measured next.

Raw results: [`rend-cache-results-64.json`](rend-cache-results-64.json) and
[`rend-cache-results-256.json`](rend-cache-results-256.json). Build hash records the
base checkout commit; the native experiment is an uncommitted source change on top.
The original prototype report remains historical and its speed ratio must not be
presented as a Cycles speedup.
