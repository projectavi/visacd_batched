#pragma once

#include <core.hpp>
#include <device_mesh.hpp>
#include <memory>
#include <vector>

namespace neural_acd {

class BatchExecutor;

struct MergeBatchInput {
  MeshList *parts = nullptr;
  MeshList *hulls = nullptr;
  std::vector<std::shared_ptr<DeviceMesh>> *part_devices = nullptr;
  std::vector<std::shared_ptr<DeviceMesh>> *hull_devices = nullptr;
  std::vector<double> *part_hausdorff = nullptr;
  size_t target_part_count = 0;
  bool use_threshold_merging = true;
  double current_concavity = 0.0;
  double threshold = 0.0;
  RandomEngine *engine = nullptr;
};

class MergeBatchRuntime {
public:
  MergeBatchRuntime();
  ~MergeBatchRuntime();

  MergeBatchRuntime(const MergeBatchRuntime &) = delete;
  MergeBatchRuntime &operator=(const MergeBatchRuntime &) = delete;
  MergeBatchRuntime(MergeBatchRuntime &&) noexcept;
  MergeBatchRuntime &operator=(MergeBatchRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void merge_convex_hulls_batch(
      const std::vector<MergeBatchInput> &, MergeBatchRuntime &, size_t,
      double, BatchExecutor *);
};

void merge_convex_hulls_batch(
    const std::vector<MergeBatchInput> &inputs,
    MergeBatchRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7, BatchExecutor *executor = nullptr);

} // namespace neural_acd
