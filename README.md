# VisACD: Visibility-Based GPU-Accelerated Approximate Convex Decomposition

[Project Page](https://3dlg-hcvc.github.io/visacd) | [Paper](https://arxiv.org/abs/2604.04244)

![Teaser](docs/visuals/teaser.png)

VisACD is a visibility-based, GPU-accelerated algorithm for intersection-free approximate convex decomposition (ACD). It is rotation-equivariant, making it robust to variations in input mesh orientation. Compared to prior work, VisACD produces decompositions that more closely approximate the original geometry while using fewer parts, and does so with significantly improved efficiency.

At the core of the approach is a **visibility edge** concavity metric: edges between pairs of vertices that lie outside the mesh without intersecting it. A convex mesh contains no such edges, while increasingly concave geometry produces more. The best cutting plane is the one that intersects the largest total length of visihttps://3dlg-hcvc.github.io/visacd/bility edges — simple, efficient, and interpretable. The algorithm is fully parallelized using NVIDIA OptiX and CUDA.

## Installation

**Requirements:** CUDA-capable GPU, NVIDIA OptiX 8.0, Python ≥ 3.8. Tested with CUDA 12.2. RTX 50-series GPUs (compute capability 12.0) require CUDA 12.8 or newer; CUDA 12.8 is recommended for this configuration.

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/3dlg-hcvc/visacd
cd visacd
```

### 2. Install NVIDIA OptiX

Download [OptiX 8.0](https://developer.nvidia.com/designworks/optix/downloads/legacy) and install it, then set (OptiX_INSTALL_DIR should contain include/ subfolder):

```bash
export OptiX_INSTALL_DIR=/path/to/optix
```

### 3. Install and select the CUDA compiler

For an RTX 50-series GPU, install CUDA 12.8 in the Conda environment:

```bash
conda activate visacd
conda install -c nvidia cuda-nvcc=12.8

# Reactivate the environment so the newly installed CUDA compiler hooks run.
# Clearing these variables also removes stale values left by another CUDA version.
conda deactivate
unset NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP
unset NVCC_APPEND_FLAGS NVCC_APPEND_FLAGS_BACKUP
conda activate visacd

export CUDACXX="$CONDA_PREFIX/bin/nvcc"
nvcc --version
```

The reported compiler version should be CUDA 12.8. The `CUDA Version` shown by `nvidia-smi` is the maximum version supported by the driver, not necessarily the version of `nvcc` used for this build.

If CUDA 12.8 is installed outside Conda instead, select its compiler explicitly:

```bash
export CUDACXX=/path/to/cuda-12.8/bin/nvcc
```

On a machine with multiple GPUs, set the PTX compute capability explicitly to avoid passing multiple values from `nvidia-smi` to the build. For an RTX 5090:

```bash
export PTX_COMPUTE=120
```

### 4. Install (might take a while)

When changing CUDA compilers, remove the previous generated build directory first so CMake does not reuse the old compiler path:

```bash
rm -rf build
python -m pip install -e .
```

## Usage

```python
import numpy as np
import trimesh
import visacd

# load any mesh trimesh supports
tm = trimesh.load("mesh.obj", force="mesh")

mesh = visacd.Mesh()
mesh.vertices = visacd.VecArray3d(tm.vertices.tolist())
mesh.triangles = visacd.make_vecarray3i(np.array(tm.faces, dtype=np.int32))

# optionally configure
visacd.config.score_mode = "concavity"
visacd.set_seed(42)

# decompose
result = visacd.process(mesh, concavity=0.04, num_parts=32)
print(f"Parts: {result.num_parts}, concavity: {result.concavity:.4f}")

# export as GLB
scene = trimesh.Scene()
for part in result.parts:
    scene.add_geometry(trimesh.Trimesh(
        vertices=np.array(list(part.vertices)),
        faces=np.array(list(part.triangles)),
        process=False,
    ))
scene.export("decomposition.glb")
```

## decompose.py

[decompose.py](decompose.py) is a ready-to-use CLI script:

```bash
# Output written to <mesh_name>_decomposed.glb next to the input file
python decompose.py data/cow.obj

# Custom output path
python decompose.py data/cow.obj -o out/cow_decomposed.glb

# Tune decomposition parameters
python decompose.py data/cow.obj --concavity 0.02 --num-parts 64
```

| Argument | Default | Description |
|---|---|---|
| `mesh` | — | Path to input mesh (any format trimesh supports) |
| `-o / --output` | `<mesh_name>_decomposed.glb` | Output GLB path |
| `--concavity` | `0.04` | Maximum concavity threshold |
| `--num-parts` | `40` | Maximum number of output parts |

## Citation

```bibtex
@inproceedings{fokin2026visacd,
  title={VisACD: Visibility-Based GPU-Accelerated Approximate Convex Decomposition},
  author={Fokin, Egor and Savva, Manolis},
  booktitle={47th Annual Conference of the European Association for Computer Graphics,
                  Eurographics 2026 - Short Papers},
  year={2026}
}
```
