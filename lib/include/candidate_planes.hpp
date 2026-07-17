#pragma once

#include <clip_batch.hpp>
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

struct SplitPlaneInput {
  const Mesh *mesh = nullptr;
  const DeviceMesh *device_mesh = nullptr;
  const unsigned int *sampled_edges = nullptr;
  size_t sample_count = 0;
  size_t max_candidates = 0;
  const std::vector<Plane> *flat_planes = nullptr;
  float flat_surface_weight = 1.0f;
  Plane *selected_plane = nullptr;
  bool *has_selected_plane = nullptr;
  std::vector<ClipTriangleData> *prepared_clip = nullptr;
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
  friend void generate_score_select_clip_batch(
      const std::vector<SplitPlaneInput> &, CandidatePlaneRuntime &, size_t,
      double);
};

void generate_candidate_planes_batch(
    const std::vector<CandidatePlaneInput> &inputs,
    CandidatePlaneRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

// Generates candidate planes, appends flat-surface planes, scores every plane,
// selects the first maximum, and prepares clipping data without round-tripping
// intermediate planes or scores through host memory.
void generate_score_select_clip_batch(
    const std::vector<SplitPlaneInput> &inputs,
    CandidatePlaneRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

} // namespace neural_acd
