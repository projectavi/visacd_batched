#pragma once

#include <batch_executor.hpp>
#include <core.hpp>
#include <memory>
#include <optixUtils.hpp>
#include <utility>
#include <vector>

namespace neural_acd {

struct RayGenData {
  const float *points;
  const unsigned int *new_mask;
  long long n_points;
  unsigned int has_mask;
  unsigned int *accepted_words;
  OptixTraversableHandle cage_gas;
  OptixTraversableHandle self_gas;
};
struct MissData {};
struct HitgroupData {
  const float *vertices;
  const uint3 *indices;
};

class OptixRuntime {
public:
  OptixRuntime();
  ~OptixRuntime();

  OptixRuntime(const OptixRuntime &) = delete;
  OptixRuntime &operator=(const OptixRuntime &) = delete;
  OptixRuntime(OptixRuntime &&) noexcept;
  OptixRuntime &operator=(OptixRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
  compute_intersection_matrices(
      const std::vector<std::pair<Mesh *, Mesh *>> &, OptixRuntime &, size_t,
      double, BatchExecutor *);
};

// Each request contains the point mesh and the cage to test it against. The
// returned edge lists have the same order as requests. Independent OptiX jobs
// are submitted concurrently in memory-aware waves.
std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
compute_intersection_matrices(
    const std::vector<std::pair<Mesh *, Mesh *>> &requests,
    OptixRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7, BatchExecutor *executor = nullptr);

} // namespace neural_acd
