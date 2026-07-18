#pragma once

#include <core.hpp>
#include <vector>

namespace neural_acd {

struct SparseSurfacePostGrid {
  const Mesh *mesh = nullptr;
  double scale = 1.0;
  double voxel_size = 1.0;
  std::vector<int> leaf_origins;
  std::vector<unsigned char> active;
  std::vector<double> values;
  std::vector<int> triangle_indices;
  std::vector<int> neighbour_indices;
  std::vector<double> neighbour_values;
  // next/previous sparse leaf along X, Y, and Z. Unlike the immediate
  // neighbours above, these links skip empty leaf-space exactly as OpenVDB's
  // exterior connectivity table does.
  std::vector<int> axis_neighbour_indices;
};

void postprocess_sparse_surface_cuda(SparseSurfacePostGrid &grid);

void postprocess_sparse_surface_cuda_batch(
    const std::vector<SparseSurfacePostGrid *> &grids);

} // namespace neural_acd
