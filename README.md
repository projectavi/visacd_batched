# VisACD Batch Engine

High-throughput, deterministic approximate convex decomposition for independent
triangle meshes on one NVIDIA GPU.

[Project page](https://3dlg-hcvc.github.io/visacd) ·
[Paper](https://arxiv.org/abs/2604.04244) ·
[License](LICENSE)

![VisACD decomposition examples](docs/visuals/teaser.png)

This repository contains the VisACD visibility-based decomposition algorithm
and a substantial systems rewrite for batched CUDA and NVIDIA OptiX execution.
The primary API is now `visacd.process_batch`: it accepts a complete,
heterogeneous batch, preserves input order, and schedules independent meshes
through a shared single-GPU pipeline. The original `visacd.process` API
remains available and uses the same batch engine with one mesh.

VisACD measures concavity through visibility edges: vertex pairs whose segment
lies outside the mesh without intersecting it. Convex geometry has no such
edges. At each decomposition step, candidate planes are evaluated by how they
interact with these edges, and the selected plane recursively partitions the
mesh.

## What changed

The batch engine is not a loop around the single-mesh implementation. It
coordinates work across meshes and decomposition stages so that irregular CPU,
CUDA, transfer, and OptiX work can overlap.

- Asynchronous per-mesh state machines replace batch-wide phase barriers.
- Candidate generation, plane scoring, deterministic argmax, and clip
  preparation are fused on the GPU.
- OptiX acceleration structures, pipelines, streams, and scratch capacity are
  persistent across work and repeated calls on the same host thread.
- CUDA streams, grow-only device buffers, asynchronous allocations, and pinned
  staging memory are reused.
- Connected-component labeling and compaction, inherited-edge projection,
  Hausdorff evaluation, flat-surface extraction, merge costs, convex-hull
  workloads, and active mesh data have batched GPU paths.
- Manifold preprocessing has exact CUDA implementations for surface
  voxelization, exterior tracing and sign propagation, narrow-band expansion,
  renormalization and trimming, and level-set meshing. Initial meshes, initial
  cages, and child cages use the same memory-aware preprocessing scheduler.
- The complete opt-in dense path retains CUDA meshing output as `DeviceMesh`
  storage, remaps it through exact connected-component ordering, appends
  visibility edges in place, and reuses resident initial and cage geometry in
  later CUDA and OptiX stages.
- Crossing-triangle records are compacted before transfer to reduce
  clip-related device-to-host traffic.
- Work-size bucketing and double-buffered waves overlap transfer and compute
  while memory-aware sizing controls VRAM pressure.
- A persistent CPU executor uses dynamic scheduling, priority work, and
  NUMA-aware work stealing where the host topology benefits from it.
- Contiguous NumPy input and output paths avoid Python list conversion.
- Seeded batches are repeatable across scheduling, wave-size, and CPU-thread
  choices.

## Execution model

```text
complete input batch
        |
        v
parallel CPU preparation
        |
        v
asynchronous pipeline coordinator
        |
        +--> OptiX visibility intersections on persistent streams
        +--> batched CUDA components, hulls, Hausdorff, surfaces, merges
        +--> fused split selection and compact clip preparation
        +--> exact child construction and resubmission
        |
        v
ordered ProcessResult list
```

The scheduler collects ready work into efficient GPU launches without forcing
all meshes to reach the same stage. Large and small requests are bucketed, then
split into memory-aware waves. Two runtime lanes per GPU work kind allow one
wave to transfer while another computes.

The engine targets the CUDA device visible to the process. It does not split one
batch over multiple GPUs. Use `CUDA_VISIBLE_DEVICES` to select the device on a
multi-GPU host.

## Requirements

- Python 3.8 or newer
- A CUDA-capable NVIDIA GPU and compatible driver
- NVIDIA OptiX 8.0 headers
- A CUDA toolkit with `nvcc`
- CMake 3.18 or newer and a C++17 compiler
- Git submodules initialized
- Network access during the first build for CMake FetchContent dependencies

The current engineering path is tested on Linux with CUDA 12.8 and OptiX 8.0.
Blackwell GPUs such as the RTX 50 series require a toolkit that supports compute
capability 12.0; CUDA 12.8 or newer is recommended for those devices.

## Build and installation

Initialize the vendored dependencies:

```bash
git submodule update --init --recursive
```

Download [NVIDIA OptiX 8.0](https://developer.nvidia.com/designworks/optix/downloads/legacy),
extract it, and point the build at the directory that contains its `include/`
directory:

```bash
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-8.0
```

Select a CUDA compiler before the first configure when multiple toolkits are
installed:

```bash
export CUDACXX=/path/to/cuda/bin/nvcc
```

For GPU or build environments where automatic PTX architecture detection is
ambiguous, set the compute capability without a decimal point:

```bash
# RTX 5090 / compute capability 12.0
export PTX_COMPUTE=120
```

Build a Release extension as an editable install:

```bash
python -m pip install -e .
```

### Reference workstation setup: RTX 5090 and Conda

The development workstation uses the `visacd` Conda environment with Python
3.11, CUDA `nvcc` 12.8.93, OptiX 8.0.0, and two RTX 5090 GPUs. The following is
the exact setup sequence used on that machine.

Activate the environment and install the CUDA compiler from the NVIDIA Conda
channel:

```bash
conda activate visacd
conda install -c nvidia cuda-nvcc=12.8
```

After installing or changing the Conda CUDA package, reactivate the environment
so its compiler hooks are regenerated. Clear hook variables first so values
from another CUDA installation cannot survive the switch:

```bash
conda deactivate
unset NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP
unset NVCC_APPEND_FLAGS NVCC_APPEND_FLAGS_BACKUP
conda activate visacd
```

Select the Conda `nvcc` explicitly and verify that it reports CUDA 12.8:

```bash
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
nvcc --version
```

On this checkout, the OptiX SDK is extracted directly under the repository
root. Point CMake at that exact directory:

```bash
export OptiX_INSTALL_DIR="$PWD/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"
```

The workstation has two GPUs. Set the PTX target explicitly so CMake does not
turn the multiple `nvidia-smi` result lines into an invalid architecture:

```bash
export PTX_COMPUTE=120
```

Remove any build generated with a different CUDA compiler, then install:

```bash
rm -rf build
python -m pip install -e .
```

For a single-GPU batch run on this host, select one RTX 5090 at runtime:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_parallel.py -n 200 --repeats 4
```

The resulting CMake configuration should resolve the important paths as
follows:

```text
CMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc
CUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux
OptiX_INSTALL_DIR=<repository>/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
PTX architecture=sm_120
```

The generated build is stored in `build/`. Delete that generated directory
before reinstalling after changing `CUDACXX`; otherwise CMake can retain the
previous compiler.

The CUDA version reported by `nvidia-smi` is the maximum supported by the
driver. Check `nvcc --version` to verify the toolkit that compiles VisACD.

## Batch-first quick start

The NumPy constructor is the preferred mesh input path. Vertices have shape
`(N, 3)` and are converted to `float64`; triangle indices have shape
`(M, 3)` and are converted to `int32`.

```python
from pathlib import Path

import numpy as np
import trimesh
import visacd


def load_mesh(path: Path) -> visacd.Mesh:
    source = trimesh.load(path, force="mesh")
    vertices = np.ascontiguousarray(source.vertices, dtype=np.float64)
    triangles = np.ascontiguousarray(source.faces, dtype=np.int32)
    return visacd.Mesh(vertices, triangles)


paths = [
    Path("data/samples/cow.obj"),
    Path("data/samples/teapot.obj"),
    Path("data/samples/fandisk.obj"),
]
meshes = [load_mesh(path) for path in paths]

# Automatic host-thread and GPU-wave sizing are the recommended defaults.
visacd.config.batch_cpu_threads = 0
visacd.config.max_batch_size = 0
visacd.config.batch_memory_fraction = 0.7

# Seed immediately before the complete batch.
visacd.set_seed(42)
results = visacd.process_batch(
    meshes,
    concavity=0.04,
    num_parts=32,
)

for path, result in zip(paths, results):
    print(path, result.num_parts, result.concavity)
    for part in result.parts:
        vertices = np.asarray(part.vertices)
        triangles = np.asarray(part.triangles)
```

The returned list contains one `ProcessResult` for every input mesh in the
same order. Meshes may have different vertex and triangle counts, but one call
uses common `concavity` and `num_parts` settings.

The `Mesh(vertices, triangles)` constructor copies contiguous array data into
extension-owned storage. `np.asarray(part.vertices)` and
`np.asarray(part.triangles)` expose buffer views of result storage without a
Python element-by-element conversion.

## Single-mesh compatibility

`process` remains useful for scripts that only have one mesh. Internally it
submits a one-item batch:

```python
visacd.set_seed(42)
result = visacd.process(mesh, concavity=0.04, num_parts=32)
```

For sustained throughput, assemble all available meshes and call
`process_batch` once.

## Public Python API

### `Mesh(vertices, triangles)`

Constructs an owned triangle mesh from array-like data with shape `(N, 3)` and
`(M, 3)`. The no-argument constructor and writable `vertices` and `triangles`
properties remain available for compatibility.

### `process_batch(meshes, concavity, num_parts)`

Decomposes every non-empty triangle mesh in one coordinated call. `concavity`
must be finite and non-negative. In concavity scoring mode it is the early-stop
threshold. `num_parts` must be at least one and controls the split-iteration
budget; connected-component separation and optional merging mean that the
final count is not always a strict copy of the requested value. An empty input
batch returns an empty result list.

### `process(mesh, concavity, num_parts)`

Compatibility wrapper for a one-item batch. It returns one `ProcessResult`
rather than a list.

### `set_seed(seed)`

Seeds the next decomposition call. Call it immediately before `process_batch`
when deterministic batch repeatability is required.

### `release_gpu_resources()`

Releases reusable CUDA and OptiX state owned by VisACD. Call it on the same
thread that submitted the retained batch, with no decomposition active. The
function waits for VisACD work to quiesce before destroying its buffers,
streams, OptiX objects, worker runtimes, and private CUDA memory pool.

### `ProcessResult`

| Field | Meaning |
|---|---|
| `parts` | Output meshes. These are convex hulls by default or clipped parts when `config.return_parts` is enabled. |
| `num_parts` | Actual number of meshes in `parts`. |
| `concavity` | Final measured decomposition concavity. |

## Batch behavior and repeatability

- The complete batch must be supplied up front; this API is not a streaming
  queue.
- A seeded batch is deterministic for the same build, inputs, configuration,
  and numerical environment.
- Each mesh receives an independent deterministic random stream, so worker
  scheduling does not change the result.
- Forced one-item GPU waves, automatic waves, a single CPU worker, and
  low-memory fallback paths are tested for matching output digests.
- A seeded batch is not required to match a sequence of separately seeded
  `process` calls because the random streams are assigned per batch.
- Cross-device or cross-toolkit bitwise identity is not promised; CUDA, OptiX,
  compiler, and floating-point differences can affect numerical execution.
- Configuration and seed state are process-global. Configure and seed a batch
  before submission rather than mutating them while work is running.

There is no 200-mesh input limit. Automatic CPU scheduling uses at most one
worker per useful batch item and hardware thread, with a safety ceiling of 200
workers. Larger input batches continue through the same scheduler and
memory-aware GPU waves.

## Configuration

Python configuration is exposed through the process-global `visacd.config`
object. Set it before submitting a batch; do not mutate it while VisACD work is
running.

### Batched implementation knobs

These settings control how the batch engine maps a complete input batch onto
host workers, memory-aware GPU waves, and reusable GPU state.

| Setting | Default | Meaning |
|---|---:|---|
| `batch_cpu_threads` | `0` | Host executor size. `0` selects `min(useful batch items, available hardware threads, 200)`; a positive value requests an explicit count, still capped by useful work. This is a worker count, not the number of meshes accepted by the API. |
| `max_batch_size` | `0` | Maximum number of ready items placed in most GPU waves. `0` leaves wave length to the memory-aware scheduler. A positive value is a cap rather than a target; memory pressure can produce smaller waves. Packed preprocessing waves are additionally capped at 200 items. |
| `batch_memory_fraction` | `0.7` | Fraction of currently free VRAM that VisACD may budget for growing a wave. Valid range: `(0, 1]`. The coordinator divides the budget conservatively among overlapping GPU host lanes; retained buffers from earlier work are accounted for separately. Lower this when another CUDA workload must keep headroom. |
| `retain_gpu_resources` | `True` | Keep streams, OptiX objects, the private asynchronous pool, and grow-only scratch capacity for faster warm batches. `False` performs automatic release before `process_batch` or `process` returns. See [GPU resource lifetime](#gpu-resource-lifetime). |

The defaults are the recommended starting point for heterogeneous batches and
machines with different CPU counts and VRAM capacities. `max_batch_size` and
`batch_memory_fraction` are independent: the scheduler stops a wave when it
reaches either the item cap or its estimated memory budget. For a GPU shared
with simulation or training, reduce `batch_memory_fraction` to reserve
in-call headroom and disable `retain_gpu_resources` to return VisACD-owned VRAM
after the call.

For example:

```python
visacd.config.batch_cpu_threads = 0       # automatic host parallelism
visacd.config.max_batch_size = 0          # automatic wave length
visacd.config.batch_memory_fraction = 0.5 # preserve half of free VRAM
visacd.config.retain_gpu_resources = False
results = visacd.process_batch(meshes, concavity=0.04, num_parts=32)
```

### Algorithm and result knobs

These settings also apply to the batch engine, but change decomposition or
returned-output behavior rather than scheduling.

| Setting | Default | Meaning |
|---|---:|---|
| `score_mode` | `"concavity"` | Split selection mode. Supported values are `"concavity"` and `"edge"`. |
| `use_flat_surfaces` | `True` | Include detected flat support surfaces in plane selection. |
| `flat_surface_min_area` | `0.1` | Minimum area used by flat-surface detection. |
| `flat_surface_k` | `2.0` | Flat-surface candidate weighting parameter. |
| `use_merging` | `False` | Run the optional post-decomposition merge pipeline. |
| `return_parts` | `False` | Return clipped part surfaces instead of final convex-hull meshes when enabled. |

### GPU resource lifetime

Retained mode is intended for a dedicated VisACD process or several adjacent
decomposition calls. It avoids recreating streams, OptiX state, and grow-only
scratch buffers, keeping later calls warm. Releasing resources makes the next
decomposition rebuild that reusable state.

For a GPU shared in phases with simulation or training, either release
automatically:

```python
visacd.config.retain_gpu_resources = False
results = visacd.process_batch(meshes, concavity=0.04, num_parts=32)
# VisACD VRAM has been released here.
```

or retain resources across a short decomposition burst and release explicitly:

```python
visacd.config.retain_gpu_resources = True
first = visacd.process_batch(first_meshes, 0.04, 32)
second = visacd.process_batch(second_meshes, 0.04, 32)
visacd.release_gpu_resources()
# Restart the simulator or RL workload here.
```

VisACD asynchronous allocations use a private CUDA pool, so releasing VisACD
does not trim a CUDA pool owned by PyTorch or another library. The release does
not call `cudaDeviceReset` and therefore does not invalidate another library's
CUDA context. CUDA code modules and driver context state can leave a small
fixed residual allocation, but reusable streams, scratch buffers, OptiX state,
and the VisACD-owned pool are released.

Automatic release serializes simultaneous top-level VisACD calls so resources
cannot be destroyed under active work. Explicit release must be called from
the thread that owns the retained resources.

### Advanced scheduler, CUDA, and OptiX knobs

These environment variables expose lower-level paths used for architecture
experiments, profiling, and regressions. Leave them at their defaults unless a
benchmark demonstrates a benefit on the deployment machine. Set allocator and
execution-path variables before the first VisACD GPU call in the process.

| Environment variable | Default | Purpose |
|---|---|---|
| `VISACD_OPTIX_BUILD_PREFERENCE` | `trace` | OptiX GAS flag: `trace`, `build`, or `none`. |
| `VISACD_OPTIX_MAX_CONCURRENCY` | memory-aware | Positive cap on simultaneously active OptiX jobs. Unset means no fixed count cap; `max_batch_size` and available VRAM still bound each wave. |
| `VISACD_WORK_STEALING` | topology-dependent | `0` disables and `1` forces cross-queue CPU work stealing. Automatic mode enables it on a multi-NUMA-node topology and pins workers to the detected nodes. |
| `VISACD_DISABLE_WORK_BUCKETING` | disabled | Set to `1` to preserve ready-queue order instead of grouping similarly sized work before GPU wave construction. Bucketing is normally beneficial for heterogeneous batches. |
| `VISACD_DISABLE_DOUBLE_BUFFERING` | disabled | Set to `1` to use one lane per GPU work kind instead of allowing two lanes to overlap transfers and kernels when mixed work sizes make overlap worthwhile. OptiX intersections already use one lane. This can reduce peak scratch capacity at the cost of overlap. |
| `VISACD_DISABLE_GPU_EDGE_PROJECTION` | disabled | Set to `1` to use the CPU fallback for inherited visibility-edge projection rather than the CUDA path when resident mesh data is available. |
| `VISACD_ENABLE_GPU_HULL_TOPOLOGY` | disabled | Set to `1` to enable the exact experimental CUDA convex-hull topology path. |
| `VISACD_DISABLE_GPU_HULL_TOPOLOGY` | disabled | Set to `1` to force the host hull-topology path even if the enable flag is present. |
| `VISACD_DISABLE_ASYNC_ALLOCATOR` | disabled | Set to `1` before first GPU use to bypass VisACD's private CUDA memory pool and use direct allocation/free. This is primarily an allocator compatibility and lifecycle diagnostic. |

The full GPU hull-topology implementation is disabled by default because it was
neutral or slower end to end on the reference GPU when competing with the
other CUDA stages. It remains available for profiling and for larger future
devices where that balance may change.

### CUDA manifold-preprocessing controls

CUDA preprocessing is split into independently selectable exact stages. The
surface voxelizer is the throughput-positive path on the reference machine;
the remaining implementations are retained for profiling and future GPUs but
are disabled unless explicitly requested because they were neutral or slower
end to end.

| Environment variable | Default | Purpose |
|---|---|---|
| `VISACD_ENABLE_CUDA_PREPROCESS` | enabled | Batch surface voxelization for initial meshes, initial cages, and child cages. Set to `0` to select the OpenVDB reference path. Unsupported or over-budget waves fall back automatically. |
| `VISACD_DISABLE_CUDA_PREPROCESS` | disabled | Set to `1` for an explicit CPU/OpenVDB preprocessing run. An explicitly set `VISACD_ENABLE_CUDA_PREPROCESS` takes precedence. |
| `VISACD_VERIFY_CUDA_PREPROCESS` | disabled | Compute both complete CUDA-candidate and CPU-reference outputs, accept the candidate only on bit-exact equality, and otherwise use the reference. This intentionally disables packed scheduling. |
| `VISACD_PREPROCESS_WAVE_SIZE` | `200` | Maximum surface-voxelization items per packed preprocessing wave when `max_batch_size` is `0`; memory availability can reduce it. Valid range: `[1, 200]`. A positive `max_batch_size` takes precedence and is itself capped at 200 for preprocessing. |
| `VISACD_ENABLE_CUDA_PREPROCESS_SIGN` | disabled | Enable exact CUDA exterior tracing, surface cleanup, leaf sign propagation, internal-node flood, and root flood. |
| `VISACD_ENABLE_CUDA_PREPROCESS_EXPAND` | disabled | Enable the exact sparse CUDA narrow-band expansion implementation. |
| `VISACD_ENABLE_CUDA_PREPROCESS_EXPAND_DENSE` | disabled | Enable the exact dense CUDA narrow-band expansion implementation. |
| `VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE` | disabled | Enable exact CUDA renormalization and narrow-band trimming. |
| `VISACD_ENABLE_CUDA_PREPROCESS_MESH` | disabled | Enable exact CUDA level-set meshing. |
| `VISACD_ENABLE_CUDA_PREPROCESS_RESIDENT` | disabled | With dense expansion, renormalization, and meshing enabled, pass dense data directly between CUDA stages and retain meshed initial/cage geometry for downstream CUDA and OptiX reuse. |
| `VISACD_VERIFY_CUDA_PREPROCESS_EXPAND` | disabled | Compare sparse CUDA narrow-band distances with the CPU implementation and replace a non-exact result with the CPU reference. |
| `VISACD_PREPROCESS_EXPAND_DENSE_WAIT_US` | `250` | Dense-expansion queue coalescing window in microseconds, in `[0, 100000]`. A queued group launches immediately when it reaches 32 items; otherwise the worker waits this long before taking up to 200. |

`VISACD_ENABLE_CUDA_PREPROCESS_RESIDENT` only takes effect when dense expansion,
renormalization, and CUDA meshing are all enabled. Feature and trace flags use
the usual convention that unset, empty, or `0` is disabled and another
non-empty value is enabled.

### Diagnostics and validation knobs

Diagnostics print to the console and can perturb timing, so disable them for
reported throughput measurements.

| Environment variable | Output or check |
|---|---|
| `VISACD_STAGE_TIMING` | Accumulated native stage calls, item counts, and elapsed times for one top-level batch. |
| `VISACD_PREPROCESS_TRACE` | Per-call preprocessing topology, timings, hashes, and CUDA fallback reasons. |
| `VISACD_PREPROCESS_SIGN_TRACE` | CUDA sign-stage timings and fallback reasons. |
| `VISACD_PREPROCESS_EXPAND_TRACE` | Sparse/dense narrow-band expansion timings and fallback reasons. |
| `VISACD_PREPROCESS_RENORMALIZE_TRACE` | Renormalization timings and fallback reasons. |
| `VISACD_PREPROCESS_MESH_TRACE` | CUDA level-set meshing timings and fallback reasons. |
| `VISACD_HULL_TOPOLOGY_DIAGNOSTICS` | Experimental CUDA hull-topology wave timings and fallback statistics. Enable by setting the variable; disable by unsetting it. |
| `VISACD_CLIP_COMPACTION_DIAGNOSTICS` | Crossing-triangle compaction counts and device-to-host byte totals. Enable by setting the variable; disable by unsetting it. |

For the currently recommended accelerated preprocessing path:

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmark_parallel.py -n 200 --repeats 4
```

For a CPU/OpenVDB comparison run:

```bash
VISACD_DISABLE_CUDA_PREPROCESS=1 \
CUDA_VISIBLE_DEVICES=0 \
python benchmark_parallel.py -n 200 --repeats 4
```

For equivalence or architecture experiments, the complete exact CUDA chain can
be selected with:

```bash
VISACD_ENABLE_CUDA_PREPROCESS=1 \
VISACD_ENABLE_CUDA_PREPROCESS_SIGN=1 \
VISACD_ENABLE_CUDA_PREPROCESS_EXPAND_DENSE=1 \
VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE=1 \
VISACD_ENABLE_CUDA_PREPROCESS_MESH=1 \
VISACD_ENABLE_CUDA_PREPROCESS_RESIDENT=1 \
CUDA_VISIBLE_DEVICES=0 \
python benchmark_parallel.py -n 200 --repeats 4
```

## Benchmarking

`benchmark_parallel.py` submits one complete batch and writes all results to
the console:

```bash
python benchmark_parallel.py -n 200 --repeats 4 --stage-timing
```

`-n` controls the number of replicated input meshes. Use `--mesh` for a
different source mesh. The script reports elapsed time, meshes per second, part
counts, a digest of every output vertex and triangle, and whether repeated runs
match.

Useful controls:

```bash
python benchmark_parallel.py \
    -n 200 \
    --mesh data/samples/cow.obj \
    --repeats 4 \
    --cpu-threads 0 \
    --max-batch-size 0 \
    --memory-fraction 0.7 \
    --optix-build-preference trace \
    --optix-max-concurrency 0 \
    --stage-timing
```

The first call initializes CUDA and OptiX. Exclude that cold repetition when
comparing steady-state performance.

### Reference engineering benchmark

The only performance baseline used here is original VisACD: 200 sequential
calls to the original `process` implementation. The final batched column is one
`process_batch` call containing the same 200 inputs. Intermediate batched
revisions and stage-ablation timings are intentionally excluded.

Results are medians of three warm repetitions with input loading and conversion
outside the timed region, `concavity=0.04`, and `num_parts=2`.

Reference system: one NVIDIA GeForce RTX 5090 selected through
`CUDA_VISIBLE_DEVICES=0`, CUDA 12.8, driver 595.71.05, and an AMD Ryzen
Threadripper PRO 9955WX with 32 hardware threads.

| 200-mesh workload | Original VisACD, sequential | Final batched VisACD | Final throughput | Speedup |
|---|---:|---:|---:|---:|
| Identical `cow.obj` meshes | 57.8767 s | 3.5276 s | 56.70 meshes/s | 16.4x |
| Deterministic smooth cow variants | 90.5219 s | 3.5522 s | 56.30 meshes/s | 25.5x |
| Distinct Thingi10K sample | 130.3480 s | 10.1113 s | 19.78 meshes/s | 12.9x |

The smooth-variant workload contains 200 unique geometry hashes with a 1.02%
median RMS vertex warp. The heterogeneous Thingi10K sample spans 11 categories
and 116--48,023 vertices per mesh. The sampling and perturbation procedures are
implemented in [benchmark_perturbed.py](benchmark_perturbed.py) and
[benchmark_thingi10k.py](benchmark_thingi10k.py).

These measurements characterize the original-to-final improvement on one
machine. Real throughput depends strongly on topology, remeshing cost,
requested part count, VRAM, CPU capacity, and the balance between CUDA and
OptiX work. The final default keeps the throughput-positive CUDA surface
voxelizer enabled; the remaining exact preprocessing stages stay opt-in because
they did not provide a consistent end-to-end benefit during development. They
reproduce the reference topology and numerical output on the validated system.

Native stage timings are accumulated task wall times. Parallel tasks overlap,
so stage values can exceed end-to-end elapsed time and must not be added
together.

## Exactness boundaries

The rewrite keeps existing numerical behavior and does not introduce
quality-changing approximations. CUDA manifold preprocessing reproduces the
OpenVDB 8.2 leaf order, signed-flood behavior, ambiguous-face decisions,
renormalization, trimming, point order, quad order, and final triangle winding.
Every CUDA stage catches unsupported cases and execution failures and returns
to the exact OpenVDB implementation. Verification mode additionally treats any
full-output mismatch as a fallback.

Some structural and downstream boundaries deliberately remain on the CPU:

- The throughput-positive surface path reads compact surface records back to
  build and prune the sparse OpenVDB topology. The opt-in dense-resident path
  avoids intermediate host reconstruction from expansion through level-set
  meshing. Its final mesh is materialized on the host for exact downstream
  child construction and the public result API while an exact device copy is
  retained in parallel. That device copy is remapped through connected
  components and reused by hull, Hausdorff, split, and OptiX cage work.
- Child cap construction uses the existing deterministic CPU constrained
  Delaunay triangulation and first-occurrence vertex ordering.
- Convex-hull topology defaults to the existing CPU path because the exact GPU
  path did not improve reference end-to-end throughput.

GPU split preparation transfers compact crossing records back for exact child
assembly, then the resulting active meshes re-enter the device-resident batch
pipeline. Replacing the remaining child-construction boundary requires an exact
deterministic GPU triangulator that reproduces current topology and ordering.

## Command-line decomposition

`decompose.py` processes one mesh and exports a GLB scene with one node per
part:

```bash
python decompose.py data/samples/cow.obj
python decompose.py data/samples/cow.obj -o /tmp/cow_decomposed.glb
python decompose.py data/samples/cow.obj \
    --concavity 0.02 \
    --num-parts 64
```

| Argument | Default | Description |
|---|---|---|
| `mesh` | required | Input path in any format supported by trimesh. |
| `-o`, `--output` | `<mesh>_decomposed.glb` | Output GLB path. |
| `--concavity` | `0.04` | Decomposition concavity target. |
| `--num-parts` | `40` | Part-count control passed to VisACD. |

## Validation

Run API validation tests without initializing a GPU:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run the CUDA and OptiX integration suite:

```bash
VISACD_RUN_GPU_TESTS=1 \
python -m unittest discover -s tests -p "test_*.py"
```

The integration suite covers:

- result order and seeded repeatability;
- automatic, forced-wave, low-memory, and single-thread equivalence;
- mixed mesh sizes and geometries;
- flat-surface and merge pipelines;
- closed, consistently wound convex output hulls;
- NumPy input constructors and output buffer views;
- invalid inputs and configuration ranges.

It also checks every CUDA preprocessing stage against the complete OpenVDB
result and compares complete seeded decomposition digests with the full CUDA
initial- and child-cage chain enabled.

Run the standalone exhaustive preprocessing verifier across every sample OBJ,
all production scale/level-set configurations, and packed CUDA waves with:

```bash
CUDA_VISIBLE_DEVICES=0 \
python verify_preprocessing_cuda.py --full-output --packed --quiet

for wave in 1 4 32 200; do
    CUDA_VISIBLE_DEVICES=0 \
    python verify_preprocessing_cuda.py \
        --packed --quiet --max-batch-size "$wave"
done
```

On the reference build this covered 33 surface cases and 44 complete-remeshing
cases with zero mismatches or unexpected fallbacks. Packed verification was
also repeated with automatic, 1, 4, 32, and 200 item waves.

Verify complete decomposition digests for every sample mesh across automatic,
1, 4, and 32 CPU workers and automatic, 1, 4, 32, and 200 item waves with:

```bash
CUDA_VISIBLE_DEVICES=0 python verify_batch_matrix.py
```

Run the same 20-case matrix through the complete resident preprocessing chain
by adding the sign, dense-expansion, renormalization, meshing, and resident
environment variables shown above. Both matrices produced zero failures and
the same all-sample reference digest on the validated system.

The current batch rewrite has also been checked with NVIDIA Compute Sanitizer
memcheck and targeted racecheck runs. A small local memcheck can be reproduced
with:

```bash
compute-sanitizer --tool memcheck \
    python benchmark_parallel.py -n 1 --repeats 1
```

To exercise the complete opt-in preprocessing chain under the sanitizer, add
the CUDA preprocessing environment variables shown above. Targeted memcheck
and racecheck runs of the packed mesher, dense-resident handoff, device
component remap, resident OptiX reuse, and complete signed-flood hierarchy
reported zero errors or hazards on the reference build.

## Troubleshooting

### The build uses the wrong CUDA toolkit

Set `CUDACXX`, delete the generated `build/` directory, and reinstall.
Confirm the selected compiler with `nvcc --version`.

### PTX compilation receives an invalid architecture

Set `PTX_COMPUTE` explicitly, especially on hosts with multiple GPU compute
capabilities. Use values such as `75`, `89`, or `120`, without `sm_` or
a decimal point.

### A large batch runs out of device memory

Lower `visacd.config.batch_memory_fraction`, or set a positive
`visacd.config.max_batch_size`. Automatic sizing uses currently free memory,
so other processes using the same GPU affect wave size.

### VRAM is needed by another GPU workload after decomposition

Set `visacd.config.retain_gpu_resources = False` before decomposition, or call
`visacd.release_gpu_resources()` on the submitting thread after the final
batch. `batch_memory_fraction` limits peak wave size but does not control
post-batch retention.

### The first benchmark repetition is slower

The first call creates CUDA and OptiX state and grows reusable buffers. Compare
later repetitions for steady-state throughput.

### An editable install imports stale code

Confirm that Python imports the extension from the current checkout. Rebuild
after C++ or CUDA changes with `python -m pip install -e .`.

## Repository layout

| Path | Purpose |
|---|---|
| `lib/src/` | C++17 orchestration, exact CPU stages, and OptiX host integration. |
| `lib/cuda/` | Batched CUDA kernels and persistent GPU runtimes. |
| `lib/ptx/` | OptiX device programs compiled to PTX. |
| `lib/include/` | Public and internal C++ headers. |
| `lib/bindings.cpp` | Python and NumPy bindings. |
| `tests/test_batch.py` | API, determinism, and GPU integration tests. |
| `benchmark_parallel.py` | Console batch throughput and digest benchmark. |
| `verify_batch_matrix.py` | All-sample CPU-thread and GPU-wave digest matrix. |
| `verify_preprocessing_cuda.py` | Exact OpenVDB surface, packed-wave, and full-remesh verifier. |
| `decompose.py` | Single-mesh GLB command-line example. |
| `data/samples/` | Representative meshes for local validation. |
| `docs/` | Original paper website and visual assets. |

`lib/3rd/` contains vendored dependencies and should remain isolated from
unrelated changes.

## Citation

If this software or the visibility-based decomposition method supports your
work, cite the VisACD paper:

```bibtex
@inproceedings{fokin2026visacd,
  title={VisACD: Visibility-Based GPU-Accelerated Approximate Convex Decomposition},
  author={Fokin, Egor and Savva, Manolis},
  booktitle={47th Annual Conference of the European Association for Computer Graphics,
             Eurographics 2026 - Short Papers},
  year={2026}
}
```

## License

VisACD is distributed under the [Apache License 2.0](LICENSE).
