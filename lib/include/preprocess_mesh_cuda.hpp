#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

class DeviceMesh;

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
  bool retain_device_mesh = false;
  double output_scale = 1.0;
  double device_memory_fraction = 0.7;
  std::shared_ptr<DeviceMesh> device_mesh;
};

void mesh_dense_volume_cuda(DenseVolumeMeshingGrid &grid);

void mesh_dense_volume_cuda_batch(
    const std::vector<DenseVolumeMeshingGrid *> &grids);

void mesh_dense_volume_cuda_device_batch(
    const std::vector<DenseVolumeMeshingGrid *> &grids,
    const unsigned char *device_active, const double *device_values,
    const std::vector<size_t> &cell_offsets, void *producer_stream);

void release_mesh_cuda_runtime();

} // namespace neural_acd
