#pragma once

#include <vector>

namespace neural_acd {

struct SparseRenormalizeGrid {
  double voxel_size = 1.0;
  std::vector<unsigned char> active;
  std::vector<double> values;
  std::vector<int> neighbour_indices;
  std::vector<double> neighbour_values;
};

void renormalize_sparse_cuda(SparseRenormalizeGrid &grid);

void renormalize_sparse_cuda_batch(
    const std::vector<SparseRenormalizeGrid *> &grids);

} // namespace neural_acd
