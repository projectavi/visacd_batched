#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

class DeviceMesh;

struct PlaneScoreInput {
  const float *planes;
  const float *points;
  const unsigned int *edges;
  float *scores;
  int num_planes;
  int num_points;
  int num_edges;
  const DeviceMesh *device_mesh = nullptr;
};

class PlaneScoringRuntime {
public:
  PlaneScoringRuntime();
  ~PlaneScoringRuntime();

  PlaneScoringRuntime(const PlaneScoringRuntime &) = delete;
  PlaneScoringRuntime &operator=(const PlaneScoringRuntime &) = delete;
  PlaneScoringRuntime(PlaneScoringRuntime &&) noexcept;
  PlaneScoringRuntime &operator=(PlaneScoringRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void classify_and_rate_planes_batch(
      const std::vector<PlaneScoreInput> &, PlaneScoringRuntime &, size_t,
      double);
};

void classify_and_rate_planes_batch(
    const std::vector<PlaneScoreInput> &inputs,
    PlaneScoringRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

void classify_and_rate_planes_batch(
    const std::vector<PlaneScoreInput> &inputs, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

} // namespace neural_acd
