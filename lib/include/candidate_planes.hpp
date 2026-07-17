#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

class DeviceMesh;

struct CandidatePlaneInput {
  const Mesh *mesh = nullptr;
  const DeviceMesh *device_mesh = nullptr;
  const unsigned int *sampled_edges = nullptr;
  size_t sample_count = 0;
  size_t max_planes = 0;
  std::vector<Plane> *planes = nullptr;
  size_t *attempts_used = nullptr;
};

class CandidatePlaneRuntime {
public:
  CandidatePlaneRuntime();
  ~CandidatePlaneRuntime();

  CandidatePlaneRuntime(const CandidatePlaneRuntime &) = delete;
  CandidatePlaneRuntime &operator=(const CandidatePlaneRuntime &) = delete;
  CandidatePlaneRuntime(CandidatePlaneRuntime &&) noexcept;
  CandidatePlaneRuntime &operator=(CandidatePlaneRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void generate_candidate_planes_batch(
      const std::vector<CandidatePlaneInput> &, CandidatePlaneRuntime &,
      size_t, double);
};

void generate_candidate_planes_batch(
    const std::vector<CandidatePlaneInput> &inputs,
    CandidatePlaneRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

} // namespace neural_acd
