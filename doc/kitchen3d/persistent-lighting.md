# Persistent lighting experiment

This is an opt-in Kitchen 3D CPU rendering experiment, not a new production GPU
renderer. It tests camera-independent reuse of diffuse surface illumination and
invalidation after edits. No claim about the unidentified demo or RX580 performance
has been established. No actual SQL or relational database is involved.

## Run it

From the Kitchen 3D source checkout:

```sh
python3 tests/python/run_kitchen3d_test.py \
  --blender '/Applications/Kitchen 3D.app/Contents/MacOS/Kitchen 3D' \
  --script tests/python/kitchen3d_persistent_lighting.py \
  --output /Users/jamiewill/Documents/Projects/build_kitchen3d_5.3/tests/persistent-lighting
```

The runner uses isolated preferences and a 180-second timeout. Output includes
`metrics.json`, eight orbit images, cached/reference/indirect images, edit images,
a log and `persistent-lighting.blend`. The test is also registered under the
`kitchen3d` CTest label for newly configured builds. The existing build was not
reconfigured or rebuilt for this experiment.

For interactive inspection, run this in Kitchen 3D's Python Console once:

```python
import sys
sys.path.insert(0, '/Users/jamiewill/Documents/Projects/blaya/scripts/modules')
import kitchen3d_lighting
kitchen3d_lighting.register()
```

Then open the generated `.blend`, select **Kitchen 3D Persistent Lighting (CPU
Experiment)** as the render engine if needed, and press F12. Render Properties
exposes connection count, patch-update budget, freeze and per-render rebuild.
Move the camera and render again after warming. Freeze stops lighting updates,
not visibility rays or scene extraction. Unfreeze after an edit to repair lighting.
With a small update budget, repeat F12 until the pending count reaches zero.
This first version has no live rendered viewport. Registration is session-only;
repeat it after restarting Kitchen 3D. `kitchen3d_lighting.unregister()` removes it.
No startup or user preference changes are installed automatically.

## Architecture inspection and integration choice

The inspected source is `BaerBonesTechnology/kitchen-3d`, in the folder `blaya`.
The separate BLAYA extension was not touched. Existing workflow/CMake/GPU-test
edits and deleted legacy macOS bundle files were preserved.

Relevant upstream components:

- `source/blender/draw/engines/eevee/eevee_lightprobe_volume.hh`: `IrradianceBake`
  already owns surfels, surfel light evaluation/propagation and probe capture.
  Its baking state is not a persistent dynamic surface cache at runtime.
- `source/blender/draw/engines/eevee/eevee_raytrace.hh`: `RayTraceBuffer` owns
  per-view radiance/variance/tile history and reprojection matrices.
- `intern/cycles/device`: CPU, CUDA, OptiX, HIP/HIPRT, Metal and oneAPI implementations.
  Source presence does not establish packaging or device support.
- `doc/python_api/examples/bpy.types.RenderEngine.1.py`: the public custom-render
  integration point used here. New source lives in `scripts/modules/kitchen3d_lighting`.
- `tests/python/CMakeLists.txt` and `run_kitchen3d_test.py`: isolated Kitchen 3D tests.

The supplied build cache enables Cycles, Metal and the Metal graphics backend;
Vulkan and GTests are disabled. Its installed bundle path is absent. Runtime tests
therefore use the installed `/Applications/Kitchen 3D.app`, whose build identity is
recorded in the measurements, loading the new Python module directly from source.
A native rebuild and full upstream test suite were not run.

The optional Python RenderEngine is the smallest independent integration point:
it provides evaluated mesh access, CPU BVH traversal and a real Render Result
without altering EEVEE, Cycles, native DNA/RNA or backend kernels. EEVEE's existing
surfel bake machinery is a plausible subsequent GPU integration point, once this
representation's quality and update costs justify that work.

## Representation and estimator

Each evaluated triangle is a constant-radiance patch. Identity is the session's
original object UID, SHA-256 of evaluated triangle connectivity, and triangle
index. Camera transforms are absent from identity. Rigid transforms retain IDs;
connectivity edits replace that object's identities. Loading, undo and redo clear
session caches. This is not a persistent on-disk bake and IDs do not survive reload.

A patch stores direct outgoing Lambertian radiance, K cosine-weighted hemisphere
samples, hit-patch identities, and ray segments including misses. Misses contribute
black. Sampling uses deterministic seeds and therefore exactly the same quadrature
for persistent and per-frame-rebuilt baselines. It does not accumulate additional
samples while stationary: warming completes the fixed sampling representation.

For each patch i, the one-bounce estimator is:

```
L_i = D_i + albedo_i / K * sum(D_hit)
D_i = albedo_i / pi * sum(point_intensity * cos(theta) / distance^2 * visibility)
```

Cosine-weighted sampling cancels the receiver cosine/pdf factors. There is no
extra source area factor for ray-sampled transfer. Only front-facing hits receive
light. This is one indirect diffuse bounce, not a converged radiosity solution.
K is fixed independently of patch count; storage is O(N*K), not an all-pairs graph.
Duplicate sampled hits preserve their Monte Carlo weights.

Every patch is seeded regardless of camera visibility. This deliberately removes
camera-dependent coverage from the initial experiment. A cold or partially updated
cache displays uncovered pixels in magenta and reports their count. It does not
hide missing data with an ambient-light fallback. This full-scene precoverage will
not scale to arbitrary scenes without adaptive allocation and eviction.

## Invalidation, scheduling and known costs

- Camera changes reuse all direct light and connections; only scene extraction,
  primary visibility and cache lookup execute.
- Light changes dirty direct illumination on all patches. Existing transport
  samples remain reusable. Material viewport-color changes dirty the affected
  object's patches and preserve their transport samples.
- Geometry edits rebuild the CPU BVH. Changed patch owners are dirty. Each other
  patch's retained rays are tested against each changed object's old/new union
  AABB. Both vacated and newly occupied space participate. Stored misses catch
  newly introduced occluders; old blocked segments catch removed blockers.
- Direct shadow segments retain the full path to the light, even when blocked.
  Hemisphere hit segments end at the hit; miss segments end at FAR=1000 units.
- The patch budget bounds lighting ray work to at most K plus the number of lights
  per processed patch. Dirty patches are processed in deterministic mesh order.
  Cancellation is checked between patches and camera rows.
- A conservative O(N*K) resolve recomputes one-bounce radiance from cached direct
  dependencies after updates. This propagates edited source illumination to
  receivers without tracing more rays. Unprocessed entries can remain stale;
  the pending count explicitly exposes this, including during freeze.

Scene extraction/fingerprinting, BVH rebuilds, segment scans and global resolves
are **not** time-budgeted. There is no spatial ray index, reverse-adjacency work
queue, cache eviction, priority aging or asynchronous upload. Those are engineering
work for a scalable renderer, not implied by the patch-ray budget. The Python
implementation intentionally measures these costs separately. Large edit bounds
can conservatively invalidate most patches. Light movement cannot assume its
influence is local with unbounded inverse-square falloff.

## Scope and explicit limitations

Supported: evaluated, opaque, flat triangles; Material viewport RGB interpreted as
linear diffuse albedo; black environment; unsoftened point lights; perspective
camera; direct plus one diffuse bounce. Subdivide large surfaces to improve patch
resolution. Normals derive from transformed triangle winding.

Shader nodes, textures, emission, smooth normals, alpha, volumes, glossy/specular
and transmission are not evaluated. The UI explicitly calls out the viewport-color
material model. Instances, non-mesh render geometry and non-point lights are
rejected. World shaders are ignored. Camera clipping, depth of field, motion blur,
render borders and output passes beyond Combined are not implemented. Fixed ray
extent (1000 units) and epsilon (1e-4) restrict scene scale. No anti-aliasing or
spatial interpolation is applied. Budgeted edits can temporarily mix old and new
lighting; freezing after edits intentionally keeps stale data.

Glossy reflection must be separate: outgoing radiance depends on view direction.
A future renderer should trace/specifically cache glossy directions (with roughness
and validity checks), optionally using the diffuse cache at later diffuse hits.
Reusing this RGB patch value as a specular solution would be incorrect.

## Measurement protocol

The fixture has a subdivided floor and three colored objects (756 initial triangle
patches), one point light, 64x64 images and 64 connections per patch. Eight camera
angles compare frozen reuse with constructing the same representation from scratch.
Tests assert zero lighting rays on orbit, identical patch values and pixel output,
and no missing covered surfaces. The two modes pay identical snapshot/visibility
costs; rebuild additionally constructs BVH and lighting data. Warm-up is separate.
Numbers are CPU wall times, including Python overhead, without GPU synchronization
or display presentation. They are not viewport FPS or RX580 measurements.

The independent reference uses actual pixel hit points and secondary-hit direct
illumination, without reading cached radiance. Combined uses 256 samples/pixel;
indirect-only uses 1024; edited scenes use 128. The reference matches the restricted
one-bounce material/light model; it is a finite-sample estimator, not ground truth
or a Cycles production comparison. Linear RGB RMSE, relative RMSE normalized by
reference RMS, and max error include the black background. Coarse patch error and
finite-sample noise are both included. The indirect-only metric prevents direct
lighting from hiding GI error.

Edits cover light motion, blocker insertion/removal from ray paths, material color,
deformation, topology subdivision, object addition and deletion. Tests compare
**every patch** with a fresh rebuild, including off-camera patches; require changed
connections for both blocker motions; and test a two-patch update budget, missing
coverage and repeated RenderEngine renders. This is evidence for these cases,
not a proof of all possible animation/dependency-graph behavior.

Memory reports both a logical packed payload estimate and a separately measured
Python `tracemalloc` peak/current allocation for the cache after edits. The latter
includes Python bookkeeping but not native BVH storage, interpreter baseline,
image allocations or GPU memory. Neither is total process memory or VRAM.

## Related work

[Radiosity](https://www.cs.helsinki.fi/group/goa/render/radios/radiosity.html)
provides the view-independent diffuse-transfer framing.
[SHARC integration](https://github.com/NVIDIA-RTX/SHARC/blob/main/docs/Integration.md)
separates sparse updates, cross-frame resolve and rendering queries; this prototype
borrows that separation conceptually, not its implementation.
[AMD Brixelizer GI](https://gpuopen.com/fidelityfx-brixelizer/) describes sparse
distance-field tracing with radiance and irradiance caches. None of these references
identifies the demo's method or verifies this prototype on an RX580.

## Next decision

Use the measured spatial error to choose finer/adaptive patches or a surface atlas
before attempting GPU acceleration. Then introduce indexed ray dependencies,
bounded maintenance, update priorities and a cache-capacity stress test. Validate
against a matching Cycles scene and animated occlusion cases at higher resolution.
A native EEVEE experiment should retain a separate glossy path and be benchmarked
on actual target hardware before making performance or compatibility claims.

## Recorded local result (2026-10-28)

Raw data is in [`persistent-lighting-results.json`](persistent-lighting-results.json).
Runtime: installed Kitchen 3D 5.3.0 Beta, build hash `4e6bc6dbbbb5`, macOS ARM64.
The Python source was loaded from this checkout; this is not a rebuild of that
binary. Eight measured orbit frames per mode, one run, no confidence interval.

| Measurement | Result |
| --- | ---: |
| Warm-up including first image | 164.4 ms |
| Frozen reuse median frame | 10.6 ms |
| Rebuild median frame | 132.9 ms |
| Lighting rays per orbit frame, reuse / rebuild | 0 / 48,600 |
| Primary camera rays per frame, both modes | 4,096 |
| Reuse vs rebuild pixel and patch difference | 0 |
| Combined linear RGB RMSE / relative RMSE | 0.05748 / 17.4% |
| Indirect-only linear RGB RMSE / relative RMSE | 0.00623 / 24.9% |
| Logical packed cache payload after edits | 1,108,320 bytes |
| Traced Python cache peak after edits | 15,518,956 bytes |

Both blocker motions invalidated 319 of 756 patches and changed 290 sampled
connection lists. Their initial dependency scans took approximately 31 ms each.
Moving the light retraced 234 shadow rays and zero transport rays. Changing the
green object's material retraced 54 shadow rays and zero transport rays. All eight
edit scenarios matched a fresh rebuild across all patches after updates drained.

These measurements show that persistent reuse works for this representation and
fixture. They also expose its limitations: the cached image has visibly faceted
shadow boundaries from centroid shading, indirect error is substantial, Python
bookkeeping is much larger than packed payload, and invalidation maintenance can
cost more than a frozen frame. This is a useful correctness baseline, not evidence
of production image quality or target-GPU performance.


## Follow-on native experiment

See [Persistent Cycles and binary .REND caches](rend-cache.md) for the subsequent
Cycles kernel integration, disk persistence, current usage and separate measurements.
The original figures above describe only the earlier Python image renderer.
