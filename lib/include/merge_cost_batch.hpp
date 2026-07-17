#pragma once

#include <cstddef>
#include <core.hpp>
#include <device_mesh.hpp>
#include <memory>
#include <vector>

namespace neural_acd {

struct MeshProximityBatchInput {
  const Mesh *first = nullptr;
  const DeviceMesh *first_device = nullptr;
  const Mesh *second = nullptr;
  const DeviceMesh *second_device = nullptr;
  double threshold = 0.0;
  bool *within_threshold = nullptr;
};

struct MeshVolumeBatchInput {
  const Mesh *mesh = nullptr;
  const DeviceMesh *device_mesh = nullptr;
  double *volume = nullptr;
};

class MergeCostBatchRuntime {
public:
  MergeCostBatchRuntime();
  ~MergeCostBatchRuntime();

  MergeCostBatchRuntime(const MergeCostBatchRuntime &) = delete;
  MergeCostBatchRuntime &
  operator=(const MergeCostBatchRuntime &) = delete;
  MergeCostBatchRuntime(MergeCostBatchRuntime &&) noexcept;
  MergeCostBatchRuntime &
  operator=(MergeCostBatchRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void evaluate_mesh_proximity_batch(
      const std::vector<MeshProximityBatchInput> &,
      MergeCostBatchRuntime &, size_t, double);
  friend void evaluate_mesh_volumes_batch(
      const std::vector<MeshVolumeBatchInput> &,
      MergeCostBatchRuntime &, size_t, double);
};

void evaluate_mesh_proximity_batch(
    const std::vector<MeshProximityBatchInput> &inputs,
    MergeCostBatchRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

void evaluate_mesh_volumes_batch(
    const std::vector<MeshVolumeBatchInput> &inputs,
    MergeCostBatchRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

} // namespace neural_acd
