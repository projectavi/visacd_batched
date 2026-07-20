#pragma once

#include <vector>

namespace neural_acd {

struct SparseHierarchyFloodGrid {
  double exterior_width = 1.0;
  double interior_width = -1.0;
  std::vector<double> leaf_first_values;
  std::vector<double> leaf_last_values;
  std::vector<int> lower_child_indices;
  std::vector<double> lower_tile_values;
  std::vector<int> upper_child_indices;
  std::vector<double> upper_tile_values;
  std::vector<int> upper_origins;
  std::vector<unsigned char> root_inside_gaps;
};

void flood_sparse_hierarchy_cuda(SparseHierarchyFloodGrid &grid);

void flood_sparse_hierarchy_cuda_batch(
    const std::vector<SparseHierarchyFloodGrid *> &grids);

void release_flood_cuda_runtime();

} // namespace neural_acd
