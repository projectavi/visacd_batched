#pragma once

#include <cstddef>
#include <vector>

namespace neural_acd {

struct PlaneScoreInput {
  const float *planes;
  const float *points;
  const unsigned int *edges;
  float *scores;
  int num_planes;
  int num_points;
  int num_edges;
};

void classify_and_rate_planes_batch(
    const std::vector<PlaneScoreInput> &inputs, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

} // namespace neural_acd
