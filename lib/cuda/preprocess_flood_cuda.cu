#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <preprocess_flood_cuda.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace neural_acd {
namespace {

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

constexpr int kLowerDimension = 16;
constexpr int kLowerSlots =
    kLowerDimension * kLowerDimension * kLowerDimension;
constexpr int kUpperDimension = 32;
constexpr int kUpperSlots =
    kUpperDimension * kUpperDimension * kUpperDimension;
constexpr int kUpperExtent = 4096;

struct PackedFloodGrid {
  size_t leaf_offset;
  size_t lower_offset;
  size_t lower_count;
  size_t upper_offset;
  size_t upper_count;
  size_t root_gap_offset;
  double exterior_width;
  double interior_width;
};

template <int Dimension>
__device__ void flood_node(
    size_t node, const int *child_indices,
    const double *child_first_values, const double *child_last_values,
    double exterior_width, double interior_width, double *tile_values,
    double *first_values, double *last_values) {
  constexpr int slots = Dimension * Dimension * Dimension;
  const size_t base = node * static_cast<size_t>(slots);
  int first_child = -1;
  for (int position = 0; position < slots; ++position) {
    if (child_indices[base + position] >= 0) {
      first_child = child_indices[base + position];
      break;
    }
  }
  if (first_child < 0) {
    first_values[node] = exterior_width;
    last_values[node] = exterior_width;
    for (int position = 0; position < slots; ++position)
      tile_values[base + position] = exterior_width;
    return;
  }

  bool x_inside = child_first_values[first_child] < 0.0;
  bool y_inside = x_inside;
  bool z_inside = x_inside;
  for (int x = 0; x < Dimension; ++x) {
    const int x00 = x * Dimension * Dimension;
    int child = child_indices[base + x00];
    if (child >= 0)
      x_inside = child_last_values[child] < 0.0;
    y_inside = x_inside;
    for (int y = 0; y < Dimension; ++y) {
      const int xy0 = x00 + y * Dimension;
      child = child_indices[base + xy0];
      if (child >= 0)
        y_inside = child_last_values[child] < 0.0;
      z_inside = y_inside;
      for (int z = 0; z < Dimension; ++z) {
        const int position = xy0 + z;
        child = child_indices[base + position];
        if (child >= 0) {
          z_inside = child_last_values[child] < 0.0;
        } else {
          tile_values[base + position] =
              z_inside ? interior_width : exterior_width;
        }
      }
    }
  }

  const int first = child_indices[base];
  first_values[node] =
      first >= 0 ? child_first_values[first] : tile_values[base];
  const int last = child_indices[base + slots - 1];
  last_values[node] =
      last >= 0 ? child_last_values[last]
                : tile_values[base + slots - 1];
}

__global__ void flood_lower_nodes_kernel(
    size_t node_count, const int *node_grid_indices,
    const PackedFloodGrid *grids, const int *child_indices,
    const double *leaf_first_values, const double *leaf_last_values,
    double *tile_values, double *first_values, double *last_values) {
  const size_t node =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node >= node_count)
    return;
  const PackedFloodGrid grid = grids[node_grid_indices[node]];
  flood_node<kLowerDimension>(
      node, child_indices, leaf_first_values, leaf_last_values,
      grid.exterior_width, grid.interior_width, tile_values,
      first_values, last_values);
}

__global__ void flood_upper_nodes_kernel(
    size_t node_count, const int *node_grid_indices,
    const PackedFloodGrid *grids, const int *child_indices,
    const double *lower_first_values, const double *lower_last_values,
    double *tile_values, double *first_values, double *last_values) {
  const size_t node =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node >= node_count)
    return;
  const PackedFloodGrid grid = grids[node_grid_indices[node]];
  flood_node<kUpperDimension>(
      node, child_indices, lower_first_values, lower_last_values,
      grid.exterior_width, grid.interior_width, tile_values,
      first_values, last_values);
}

__global__ void mark_root_inside_gaps_kernel(
    size_t grid_count, const PackedFloodGrid *grids,
    const int3 *upper_origins, const double *upper_first_values,
    const double *upper_last_values, unsigned char *inside_gaps) {
  const size_t grid_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (grid_index >= grid_count)
    return;
  const PackedFloodGrid grid = grids[grid_index];
  if (grid.upper_count < 2)
    return;
  for (size_t local = 0; local + 1 < grid.upper_count; ++local) {
    const size_t first = grid.upper_offset + local;
    const size_t second = first + 1;
    const int3 a = upper_origins[first];
    const int3 b = upper_origins[second];
    const bool separated =
        a.x == b.x && a.y == b.y && b.z - a.z != kUpperExtent;
    inside_gaps[grid.root_gap_offset + local] =
        separated && upper_last_values[first] < 0.0 &&
                upper_first_values[second] < 0.0
            ? 1
            : 0;
  }
}

struct Runtime {
  std::mutex mutex;
  cudaStream_t stream = nullptr;
  DeviceBuffer grids;
  DeviceBuffer lower_grid_indices, upper_grid_indices;
  DeviceBuffer leaf_first_values, leaf_last_values;
  DeviceBuffer lower_child_indices, lower_tile_values;
  DeviceBuffer lower_first_values, lower_last_values;
  DeviceBuffer upper_child_indices, upper_tile_values;
  DeviceBuffer upper_first_values, upper_last_values, upper_origins;
  DeviceBuffer root_inside_gaps;
  PinnedBuffer host_grids;
  PinnedBuffer host_lower_grid_indices, host_upper_grid_indices;
  PinnedBuffer host_leaf_first_values, host_leaf_last_values;
  PinnedBuffer host_lower_child_indices, host_lower_tile_values;
  PinnedBuffer host_upper_child_indices, host_upper_tile_values;
  PinnedBuffer host_upper_origins, host_root_inside_gaps;

  ~Runtime() {
    if (stream)
      cudaStreamDestroy(stream);
  }

  void ensure_stream() {
    if (!stream) {
      cuda_memory::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create sparse hierarchy flood CUDA stream");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }
};

Runtime &runtime() {
  static Runtime state;
  return state;
}

} // namespace

void flood_sparse_hierarchy_cuda(SparseHierarchyFloodGrid &grid) {
  std::vector<SparseHierarchyFloodGrid *> grids{&grid};
  flood_sparse_hierarchy_cuda_batch(grids);
}

void flood_sparse_hierarchy_cuda_batch(
    const std::vector<SparseHierarchyFloodGrid *> &grids) {
  size_t leaf_count = 0;
  size_t lower_count = 0;
  size_t upper_count = 0;
  size_t root_gap_count = 0;
  for (SparseHierarchyFloodGrid *grid : grids) {
    if (!grid || !std::isfinite(grid->exterior_width) ||
        grid->exterior_width < 0.0 ||
        !std::isfinite(grid->interior_width) ||
        grid->interior_width > 0.0 ||
        grid->leaf_first_values.size() !=
            grid->leaf_last_values.size() ||
        grid->lower_child_indices.size() % kLowerSlots != 0 ||
        grid->upper_child_indices.size() % kUpperSlots != 0 ||
        grid->upper_origins.size() % 3 != 0 ||
        grid->upper_origins.size() / 3 !=
            grid->upper_child_indices.size() / kUpperSlots) {
      throw std::invalid_argument(
          "sparse hierarchy flood grid is malformed");
    }
    const size_t leaves = grid->leaf_first_values.size();
    const size_t lowers =
        grid->lower_child_indices.size() / kLowerSlots;
    const size_t uppers =
        grid->upper_child_indices.size() / kUpperSlots;
    for (int child : grid->lower_child_indices) {
      if (child < -1 ||
          (child >= 0 && static_cast<size_t>(child) >= leaves))
        throw std::invalid_argument(
            "sparse hierarchy lower child index is invalid");
    }
    for (int child : grid->upper_child_indices) {
      if (child < -1 ||
          (child >= 0 && static_cast<size_t>(child) >= lowers))
        throw std::invalid_argument(
            "sparse hierarchy upper child index is invalid");
    }
    if ((leaves != 0 && (lowers == 0 || uppers == 0)) ||
        leaf_count > std::numeric_limits<size_t>::max() - leaves ||
        lower_count > std::numeric_limits<size_t>::max() - lowers ||
        upper_count > std::numeric_limits<size_t>::max() - uppers ||
        root_gap_count > std::numeric_limits<size_t>::max() -
                             (uppers > 0 ? uppers - 1 : 0)) {
      throw std::overflow_error("sparse hierarchy flood batch overflow");
    }
    leaf_count += leaves;
    lower_count += lowers;
    upper_count += uppers;
    root_gap_count += uppers > 0 ? uppers - 1 : 0;
    grid->lower_tile_values.resize(lowers * kLowerSlots);
    grid->upper_tile_values.resize(uppers * kUpperSlots);
    grid->root_inside_gaps.resize(uppers > 0 ? uppers - 1 : 0);
  }
  if (grids.empty() || leaf_count == 0)
    return;
  if (lower_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      upper_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("sparse hierarchy flood launch overflow");

  Runtime &state = runtime();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_stream();
#define ENSURE(buffer, bytes, message) state.buffer.ensure(bytes, message)
  ENSURE(grids, grids.size() * sizeof(PackedFloodGrid),
         "allocate sparse hierarchy grids");
  ENSURE(lower_grid_indices, lower_count * sizeof(int),
         "allocate sparse hierarchy lower owners");
  ENSURE(upper_grid_indices, upper_count * sizeof(int),
         "allocate sparse hierarchy upper owners");
  ENSURE(leaf_first_values, leaf_count * sizeof(double),
         "allocate sparse hierarchy leaf first values");
  ENSURE(leaf_last_values, leaf_count * sizeof(double),
         "allocate sparse hierarchy leaf last values");
  ENSURE(lower_child_indices, lower_count * kLowerSlots * sizeof(int),
         "allocate sparse hierarchy lower children");
  ENSURE(lower_tile_values, lower_count * kLowerSlots * sizeof(double),
         "allocate sparse hierarchy lower tiles");
  ENSURE(lower_first_values, lower_count * sizeof(double),
         "allocate sparse hierarchy lower first values");
  ENSURE(lower_last_values, lower_count * sizeof(double),
         "allocate sparse hierarchy lower last values");
  ENSURE(upper_child_indices, upper_count * kUpperSlots * sizeof(int),
         "allocate sparse hierarchy upper children");
  ENSURE(upper_tile_values, upper_count * kUpperSlots * sizeof(double),
         "allocate sparse hierarchy upper tiles");
  ENSURE(upper_first_values, upper_count * sizeof(double),
         "allocate sparse hierarchy upper first values");
  ENSURE(upper_last_values, upper_count * sizeof(double),
         "allocate sparse hierarchy upper last values");
  ENSURE(upper_origins, upper_count * sizeof(int3),
         "allocate sparse hierarchy upper origins");
  ENSURE(root_inside_gaps, root_gap_count * sizeof(unsigned char),
         "allocate sparse hierarchy root gaps");
  ENSURE(host_grids, grids.size() * sizeof(PackedFloodGrid),
         "allocate host sparse hierarchy grids");
  ENSURE(host_lower_grid_indices, lower_count * sizeof(int),
         "allocate host sparse hierarchy lower owners");
  ENSURE(host_upper_grid_indices, upper_count * sizeof(int),
         "allocate host sparse hierarchy upper owners");
  ENSURE(host_leaf_first_values, leaf_count * sizeof(double),
         "allocate host sparse hierarchy leaf first values");
  ENSURE(host_leaf_last_values, leaf_count * sizeof(double),
         "allocate host sparse hierarchy leaf last values");
  ENSURE(host_lower_child_indices,
         lower_count * kLowerSlots * sizeof(int),
         "allocate host sparse hierarchy lower children");
  ENSURE(host_lower_tile_values,
         lower_count * kLowerSlots * sizeof(double),
         "allocate host sparse hierarchy lower tiles");
  ENSURE(host_upper_child_indices,
         upper_count * kUpperSlots * sizeof(int),
         "allocate host sparse hierarchy upper children");
  ENSURE(host_upper_tile_values,
         upper_count * kUpperSlots * sizeof(double),
         "allocate host sparse hierarchy upper tiles");
  ENSURE(host_upper_origins, upper_count * sizeof(int3),
         "allocate host sparse hierarchy upper origins");
  ENSURE(host_root_inside_gaps,
         root_gap_count * sizeof(unsigned char),
         "allocate host sparse hierarchy root gaps");
#undef ENSURE

  auto *host_grids = state.host_grids.as<PackedFloodGrid>();
  int *host_lower_owners = state.host_lower_grid_indices.as<int>();
  int *host_upper_owners = state.host_upper_grid_indices.as<int>();
  double *host_leaf_first =
      state.host_leaf_first_values.as<double>();
  double *host_leaf_last = state.host_leaf_last_values.as<double>();
  int *host_lower_children =
      state.host_lower_child_indices.as<int>();
  int *host_upper_children =
      state.host_upper_child_indices.as<int>();
  int3 *host_upper_origins = state.host_upper_origins.as<int3>();
  size_t leaf_offset = 0, lower_offset = 0, upper_offset = 0;
  size_t root_gap_offset = 0;
  for (size_t grid_index = 0; grid_index < grids.size(); ++grid_index) {
    SparseHierarchyFloodGrid &grid = *grids[grid_index];
    const size_t leaves = grid.leaf_first_values.size();
    const size_t lowers = grid.lower_child_indices.size() / kLowerSlots;
    const size_t uppers = grid.upper_child_indices.size() / kUpperSlots;
    host_grids[grid_index] =
        {leaf_offset, lower_offset, lowers, upper_offset, uppers,
         root_gap_offset, grid.exterior_width, grid.interior_width};
    std::fill_n(host_lower_owners + lower_offset, lowers,
                static_cast<int>(grid_index));
    std::fill_n(host_upper_owners + upper_offset, uppers,
                static_cast<int>(grid_index));
    std::copy(grid.leaf_first_values.begin(),
              grid.leaf_first_values.end(), host_leaf_first + leaf_offset);
    std::copy(grid.leaf_last_values.begin(),
              grid.leaf_last_values.end(), host_leaf_last + leaf_offset);
    for (size_t index = 0; index < grid.lower_child_indices.size();
         ++index) {
      const int child = grid.lower_child_indices[index];
      host_lower_children[lower_offset * kLowerSlots + index] =
          child < 0 ? -1 : static_cast<int>(leaf_offset) + child;
    }
    for (size_t index = 0; index < grid.upper_child_indices.size();
         ++index) {
      const int child = grid.upper_child_indices[index];
      host_upper_children[upper_offset * kUpperSlots + index] =
          child < 0 ? -1 : static_cast<int>(lower_offset) + child;
    }
    for (size_t index = 0; index < uppers; ++index) {
      host_upper_origins[upper_offset + index] = make_int3(
          grid.upper_origins[index * 3],
          grid.upper_origins[index * 3 + 1],
          grid.upper_origins[index * 3 + 2]);
    }
    leaf_offset += leaves;
    lower_offset += lowers;
    upper_offset += uppers;
    root_gap_offset += uppers > 0 ? uppers - 1 : 0;
  }

  const auto copy_to_device = [&](void *destination, const void *source,
                                  size_t bytes, const char *message) {
    if (bytes == 0)
      return;
    cuda_memory::check(
        cudaMemcpyAsync(destination, source, bytes,
                        cudaMemcpyHostToDevice, state.stream),
        message);
  };
#define COPY(buffer, type, source, bytes, message) \
  copy_to_device(state.buffer.as<type>(), source, bytes, message)
  COPY(grids, PackedFloodGrid, host_grids,
       grids.size() * sizeof(PackedFloodGrid),
       "copy sparse hierarchy grids");
  COPY(lower_grid_indices, int, host_lower_owners,
       lower_count * sizeof(int), "copy sparse hierarchy lower owners");
  COPY(upper_grid_indices, int, host_upper_owners,
       upper_count * sizeof(int), "copy sparse hierarchy upper owners");
  COPY(leaf_first_values, double, host_leaf_first,
       leaf_count * sizeof(double),
       "copy sparse hierarchy leaf first values");
  COPY(leaf_last_values, double, host_leaf_last,
       leaf_count * sizeof(double),
       "copy sparse hierarchy leaf last values");
  COPY(lower_child_indices, int, host_lower_children,
       lower_count * kLowerSlots * sizeof(int),
       "copy sparse hierarchy lower children");
  COPY(upper_child_indices, int, host_upper_children,
       upper_count * kUpperSlots * sizeof(int),
       "copy sparse hierarchy upper children");
  COPY(upper_origins, int3, host_upper_origins,
       upper_count * sizeof(int3),
       "copy sparse hierarchy upper origins");
#undef COPY

  constexpr int threads = 128;
  const int lower_blocks =
      static_cast<int>((lower_count + threads - 1) / threads);
  const int upper_blocks =
      static_cast<int>((upper_count + threads - 1) / threads);
  const int grid_blocks =
      static_cast<int>((grids.size() + threads - 1) / threads);
  flood_lower_nodes_kernel<<<lower_blocks, threads, 0, state.stream>>>(
      lower_count, state.lower_grid_indices.as<int>(),
      state.grids.as<PackedFloodGrid>(),
      state.lower_child_indices.as<int>(),
      state.leaf_first_values.as<double>(),
      state.leaf_last_values.as<double>(),
      state.lower_tile_values.as<double>(),
      state.lower_first_values.as<double>(),
      state.lower_last_values.as<double>());
  cuda_memory::check(cudaGetLastError(),
                     "launch sparse hierarchy lower flood");
  flood_upper_nodes_kernel<<<upper_blocks, threads, 0, state.stream>>>(
      upper_count, state.upper_grid_indices.as<int>(),
      state.grids.as<PackedFloodGrid>(),
      state.upper_child_indices.as<int>(),
      state.lower_first_values.as<double>(),
      state.lower_last_values.as<double>(),
      state.upper_tile_values.as<double>(),
      state.upper_first_values.as<double>(),
      state.upper_last_values.as<double>());
  cuda_memory::check(cudaGetLastError(),
                     "launch sparse hierarchy upper flood");
  if (root_gap_count > 0) {
    mark_root_inside_gaps_kernel<<<grid_blocks, threads, 0, state.stream>>>(
        grids.size(), state.grids.as<PackedFloodGrid>(),
        state.upper_origins.as<int3>(),
        state.upper_first_values.as<double>(),
        state.upper_last_values.as<double>(),
        state.root_inside_gaps.as<unsigned char>());
    cuda_memory::check(cudaGetLastError(),
                       "launch sparse hierarchy root flood");
  }

  cuda_memory::check(
      cudaMemcpyAsync(state.host_lower_tile_values.as<double>(),
                      state.lower_tile_values.as<double>(),
                      lower_count * kLowerSlots * sizeof(double),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy sparse hierarchy lower tiles");
  cuda_memory::check(
      cudaMemcpyAsync(state.host_upper_tile_values.as<double>(),
                      state.upper_tile_values.as<double>(),
                      upper_count * kUpperSlots * sizeof(double),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy sparse hierarchy upper tiles");
  if (root_gap_count > 0) {
    cuda_memory::check(
        cudaMemcpyAsync(state.host_root_inside_gaps.as<unsigned char>(),
                        state.root_inside_gaps.as<unsigned char>(),
                        root_gap_count * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost, state.stream),
        "copy sparse hierarchy root gaps");
  }
  cuda_memory::check(cudaStreamSynchronize(state.stream),
                     "wait for sparse hierarchy flood");

  lower_offset = 0;
  upper_offset = 0;
  root_gap_offset = 0;
  for (SparseHierarchyFloodGrid *grid : grids) {
    const size_t lowers = grid->lower_child_indices.size() / kLowerSlots;
    const size_t uppers = grid->upper_child_indices.size() / kUpperSlots;
    std::copy_n(state.host_lower_tile_values.as<double>() +
                    lower_offset * kLowerSlots,
                grid->lower_tile_values.size(),
                grid->lower_tile_values.begin());
    std::copy_n(state.host_upper_tile_values.as<double>() +
                    upper_offset * kUpperSlots,
                grid->upper_tile_values.size(),
                grid->upper_tile_values.begin());
    if (!grid->root_inside_gaps.empty()) {
      std::copy_n(state.host_root_inside_gaps.as<unsigned char>() +
                      root_gap_offset,
                  grid->root_inside_gaps.size(),
                  grid->root_inside_gaps.begin());
    }
    lower_offset += lowers;
    upper_offset += uppers;
    root_gap_offset += uppers > 0 ? uppers - 1 : 0;
  }
}

} // namespace neural_acd
