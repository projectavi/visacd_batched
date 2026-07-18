#pragma once

#include <core.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace neural_acd {

struct SurfaceVoxelRecord {
  int x = 0;
  int y = 0;
  int z = 0;
  double squared_distance = 0.0;
  int triangle_index = -1;
};

struct SurfaceVoxelizationResult {
  std::vector<SurfaceVoxelRecord> records;
  bool supported = false;
  std::string fallback_reason;
  size_t candidate_voxels = 0;
  double elapsed_ms = 0.0;
};

struct SurfaceVoxelizationInput {
  const Mesh *mesh = nullptr;
  double scale = 1.0;
  SurfaceVoxelizationResult *result = nullptr;
  std::function<void()> completion;
};

class ManifoldCudaRuntime {
public:
  ManifoldCudaRuntime();
  ~ManifoldCudaRuntime();

  ManifoldCudaRuntime(const ManifoldCudaRuntime &) = delete;
  ManifoldCudaRuntime &operator=(const ManifoldCudaRuntime &) = delete;
  ManifoldCudaRuntime(ManifoldCudaRuntime &&) noexcept;
  ManifoldCudaRuntime &operator=(ManifoldCudaRuntime &&) noexcept;

  SurfaceVoxelizationResult voxelize_surface(const Mesh &mesh, double scale,
                                              double memory_fraction = 0.7);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

class ManifoldCudaBatchRuntime {
public:
  ManifoldCudaBatchRuntime();
  ~ManifoldCudaBatchRuntime();

  ManifoldCudaBatchRuntime(const ManifoldCudaBatchRuntime &) = delete;
  ManifoldCudaBatchRuntime &
  operator=(const ManifoldCudaBatchRuntime &) = delete;
  ManifoldCudaBatchRuntime(ManifoldCudaBatchRuntime &&) noexcept;
  ManifoldCudaBatchRuntime &
  operator=(ManifoldCudaBatchRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void voxelize_surfaces_batch(
      const std::vector<SurfaceVoxelizationInput> &,
      ManifoldCudaBatchRuntime &, size_t, double);
};

void voxelize_surfaces_batch(
    const std::vector<SurfaceVoxelizationInput> &inputs,
    ManifoldCudaBatchRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

std::vector<SurfaceVoxelRecord>
reference_surface_voxelization(const Mesh &mesh, double scale);

} // namespace neural_acd
