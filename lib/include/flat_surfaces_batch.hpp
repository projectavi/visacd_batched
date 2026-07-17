#pragma once

#include <cstddef>
#include <memory>
#include <support_surface.hpp>
#include <vector>

namespace neural_acd {

class BatchExecutor;

struct FlatSurfaceBatchInput {
  const Mesh *mesh = nullptr;
  double min_area = 0.0;
  std::vector<Surface> *surfaces = nullptr;
};

class FlatSurfaceBatchRuntime {
public:
  FlatSurfaceBatchRuntime();
  ~FlatSurfaceBatchRuntime();

  FlatSurfaceBatchRuntime(const FlatSurfaceBatchRuntime &) = delete;
  FlatSurfaceBatchRuntime &
  operator=(const FlatSurfaceBatchRuntime &) = delete;
  FlatSurfaceBatchRuntime(FlatSurfaceBatchRuntime &&) noexcept;
  FlatSurfaceBatchRuntime &
  operator=(FlatSurfaceBatchRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void extract_flat_surfaces_batch(
      const std::vector<FlatSurfaceBatchInput> &,
      FlatSurfaceBatchRuntime &, size_t, double, BatchExecutor *);
};

void extract_flat_surfaces_batch(
    const std::vector<FlatSurfaceBatchInput> &inputs,
    FlatSurfaceBatchRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7, BatchExecutor *executor = nullptr);

} // namespace neural_acd
