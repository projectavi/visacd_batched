#pragma once

#include <array>
#include <cstddef>
#include <core.hpp>
#include <memory>
#include <vector>

namespace neural_acd {

constexpr size_t kHausdorffCandidateCount = 10;

struct PreparedHausdorffDirection {
  const Mesh *target = nullptr;
  std::vector<Vec3D> queries;
  std::vector<std::array<int, kHausdorffCandidateCount>> candidate_triangles;
  std::vector<unsigned char> candidate_counts;
  std::vector<double> nearest_sample_distance_squared;
};

struct PreparedHausdorffJob {
  std::array<PreparedHausdorffDirection, 2> directions;
  double result = INF;
  bool valid = false;
};

PreparedHausdorffJob prepare_hausdorff_job(
    Mesh &first, Mesh &second, unsigned int resolution, bool flag,
    RandomEngine &engine);

class HausdorffRuntime {
public:
  HausdorffRuntime();
  ~HausdorffRuntime();

  HausdorffRuntime(const HausdorffRuntime &) = delete;
  HausdorffRuntime &operator=(const HausdorffRuntime &) = delete;
  HausdorffRuntime(HausdorffRuntime &&) noexcept;
  HausdorffRuntime &operator=(HausdorffRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void evaluate_hausdorff_batch(
      const std::vector<PreparedHausdorffJob *> &, HausdorffRuntime &, size_t,
      double);
};

void evaluate_hausdorff_batch(
    const std::vector<PreparedHausdorffJob *> &jobs,
    HausdorffRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

} // namespace neural_acd
