#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

class DeviceMesh;

struct NarrowbandFragment {
  int triangle_index = 0;
  int x = 0;
  int y = 0;
  int z = 0;
};

struct NarrowbandCandidate {
  int x = 0;
  int y = 0;
  int z = 0;
  int manhattan_limit = 0;
  size_t fragment_offset = 0;
  size_t fragment_count = 0;
};

struct NarrowbandDistance {
  double distance = 0.0;
  int triangle_index = 0;
};

struct NarrowbandEvaluationInput {
  const Mesh *mesh = nullptr;
  double scale = 1.0;
  double voxel_size = 1.0;
  const std::vector<NarrowbandFragment> *fragments = nullptr;
  const std::vector<NarrowbandCandidate> *candidates = nullptr;
  std::vector<NarrowbandDistance> *distances = nullptr;
};

struct DenseNarrowbandGrid {
  int minimum[3] = {0, 0, 0};
  int dimensions[3] = {0, 0, 0};
  double exterior_width = 0.0;
  double interior_width = 0.0;
  double voxel_size = 1.0;
  unsigned int iterations = 0;
  bool renormalize = false;
  bool mesh_output = false;
  double isovalue = 0.0;
  std::vector<unsigned char> active;
  std::vector<unsigned char> inside;
  std::vector<double> distances;
  std::vector<int> triangle_indices;
  std::vector<int> leaf_order;
  std::vector<float> points;
  std::vector<int> quads;
  bool retain_device_mesh = false;
  double output_scale = 1.0;
  double device_memory_fraction = 0.7;
  std::shared_ptr<DeviceMesh> device_mesh;
};

struct DenseNarrowbandInput {
  const Mesh *mesh = nullptr;
  double scale = 1.0;
  DenseNarrowbandGrid *grid = nullptr;
};

void expand_narrowband_dense_cuda_batch(
    const std::vector<DenseNarrowbandInput> &inputs);

void expand_narrowband_dense_cuda(
    const Mesh &mesh, double scale, DenseNarrowbandGrid &grid);

void evaluate_narrowband_distances_cuda_batch(
    const std::vector<NarrowbandEvaluationInput> &inputs);

void evaluate_narrowband_distances_cuda(
    const Mesh &mesh, double scale, double voxel_size,
    const std::vector<NarrowbandFragment> &fragments,
    const std::vector<NarrowbandCandidate> &candidates,
    std::vector<NarrowbandDistance> &distances);

void release_expand_cuda_runtime();

} // namespace neural_acd
