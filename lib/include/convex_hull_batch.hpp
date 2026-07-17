#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

class DeviceMesh;

struct ConvexHullBatchInput {
  const Mesh *mesh = nullptr;
  const DeviceMesh *device_mesh = nullptr;
  Mesh *hull = nullptr;
  bool fix_normals = true;
};

class ConvexHullBatchRuntime {
public:
  ConvexHullBatchRuntime();
  ~ConvexHullBatchRuntime();

  ConvexHullBatchRuntime(const ConvexHullBatchRuntime &) = delete;
  ConvexHullBatchRuntime &
  operator=(const ConvexHullBatchRuntime &) = delete;
  ConvexHullBatchRuntime(ConvexHullBatchRuntime &&) noexcept;
  ConvexHullBatchRuntime &
  operator=(ConvexHullBatchRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void compute_convex_hulls_batch(
      const std::vector<ConvexHullBatchInput> &,
      ConvexHullBatchRuntime &, size_t, double);
};

void compute_convex_hulls_batch(
    const std::vector<ConvexHullBatchInput> &inputs,
    ConvexHullBatchRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

} // namespace neural_acd
