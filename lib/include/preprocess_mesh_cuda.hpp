#pragma once

#include <vector>

namespace neural_acd {

struct DenseVolumeMeshingGrid {
  int minimum[3] = {0, 0, 0};
  int dimensions[3] = {0, 0, 0};
  double isovalue = 0.0;
  std::vector<unsigned char> active;
  std::vector<double> values;
  // Dense leaf indices in the exact order produced by OpenVDB tree
  // traversal. Empty leaves are allowed and do not affect output offsets.
  std::vector<int> leaf_order;
  std::vector<float> points;
  std::vector<int> quads;
};

void mesh_dense_volume_cuda(DenseVolumeMeshingGrid &grid);

void mesh_dense_volume_cuda_batch(
    const std::vector<DenseVolumeMeshingGrid *> &grids);

} // namespace neural_acd
