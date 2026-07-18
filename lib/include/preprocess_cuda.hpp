#pragma once

#include <core.hpp>
#include <cstddef>
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

std::vector<SurfaceVoxelRecord>
reference_surface_voxelization(const Mesh &mesh, double scale);

} // namespace neural_acd
