#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <preprocess_renormalize_cuda.hpp>
#include <resettable_runtime.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace neural_acd {
namespace {

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

constexpr size_t kLeafSize = 8 * 8 * 8;
constexpr size_t kNeighbourCount = 6;

struct PackedRenormalizeGrid {
  size_t leaf_offset;
  size_t leaf_count;
  double voxel_size;
  double exterior_width;
  double interior_width;
  int trim_narrow_band;
};

__device__ double maximum(double first, double second) {
  return first < second ? second : first;
}

__device__ double minimum(double first, double second) {
  return second < first ? second : first;
}

__device__ double square(double value) { return value * value; }

__device__ double offset_value(size_t cell, const double *values,
                               const unsigned char *active,
                               double offset) {
  const double value = values[cell];
  return active[cell] ? value - offset : value;
}

__device__ double neighbour_value(
    size_t leaf, int x, int y, int z, int direction,
    const int *neighbour_indices, const double *neighbour_values,
    const double *values, const unsigned char *active, double offset) {
  int adjacent_x = x;
  int adjacent_y = y;
  int adjacent_z = z;
  bool outside = false;
  if (direction == 0) {
    outside = x == 0;
    adjacent_x = outside ? 7 : x - 1;
  } else if (direction == 1) {
    outside = x == 7;
    adjacent_x = outside ? 0 : x + 1;
  } else if (direction == 2) {
    outside = y == 0;
    adjacent_y = outside ? 7 : y - 1;
  } else if (direction == 3) {
    outside = y == 7;
    adjacent_y = outside ? 0 : y + 1;
  } else if (direction == 4) {
    outside = z == 0;
    adjacent_z = outside ? 7 : z - 1;
  } else {
    outside = z == 7;
    adjacent_z = outside ? 0 : z + 1;
  }

  size_t adjacent_leaf = leaf;
  if (outside) {
    const size_t neighbour = leaf * kNeighbourCount + direction;
    const int packed_leaf = neighbour_indices[neighbour];
    if (packed_leaf < 0)
      return neighbour_values[neighbour];
    adjacent_leaf = static_cast<size_t>(packed_leaf);
  }
  const size_t local =
      (static_cast<size_t>(adjacent_x) * 8 + adjacent_y) * 8 +
      adjacent_z;
  return offset_value(adjacent_leaf * kLeafSize + local, values,
                      active, offset);
}

__global__ void renormalize_sparse_kernel(
    const PackedRenormalizeGrid *grids, const int *leaf_grid_indices,
    size_t cell_count, const int *neighbour_indices,
    const double *neighbour_values, const unsigned char *active,
    const double *values, double *output,
    unsigned char *output_active) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count)
    return;
  if (!active[cell]) {
    output[cell] = values[cell];
    output_active[cell] = 0;
    return;
  }
  const size_t leaf = cell / kLeafSize;
  const PackedRenormalizeGrid grid =
      grids[leaf_grid_indices[leaf]];
  const size_t local = cell - leaf * kLeafSize;
  const int x = static_cast<int>(local / 64);
  const int y = static_cast<int>((local / 8) & 7);
  const int z = static_cast<int>(local & 7);
  const double dx = grid.voxel_size;
  const double inverse_dx = 1.0 / dx;
  const double offset = 0.8 * dx;
  const double phi0 = values[cell] - offset;

  const double down_x =
      phi0 - neighbour_value(leaf, x, y, z, 0, neighbour_indices,
                             neighbour_values, values, active, offset);
  const double up_x =
      neighbour_value(leaf, x, y, z, 1, neighbour_indices,
                      neighbour_values, values, active, offset) -
      phi0;
  const double down_y =
      phi0 - neighbour_value(leaf, x, y, z, 2, neighbour_indices,
                             neighbour_values, values, active, offset);
  const double up_y =
      neighbour_value(leaf, x, y, z, 3, neighbour_indices,
                      neighbour_values, values, active, offset) -
      phi0;
  const double down_z =
      phi0 - neighbour_value(leaf, x, y, z, 4, neighbour_indices,
                             neighbour_values, values, active, offset);
  const double up_z =
      neighbour_value(leaf, x, y, z, 5, neighbour_indices,
                      neighbour_values, values, active, offset) -
      phi0;

  const double zero = 0.0;
  double norm_squared = 0.0;
  if (phi0 > 0.0) {
    norm_squared =
        maximum(square(maximum(down_x, zero)),
                square(minimum(up_x, zero)));
    norm_squared +=
        maximum(square(maximum(down_y, zero)),
                square(minimum(up_y, zero)));
    norm_squared +=
        maximum(square(maximum(down_z, zero)),
                square(minimum(up_z, zero)));
  } else {
    norm_squared =
        maximum(square(minimum(down_x, zero)),
                square(maximum(up_x, zero)));
    norm_squared +=
        maximum(square(minimum(down_y, zero)),
                square(maximum(up_y, zero)));
    norm_squared +=
        maximum(square(minimum(down_z, zero)),
                square(maximum(up_z, zero)));
  }
  const double difference =
      sqrt(norm_squared) * inverse_dx - 1.0;
  const double sign =
      phi0 / sqrt(phi0 * phi0 + norm_squared);
  const double updated = phi0 - dx * sign * difference;
  double result = minimum(phi0, updated) + offset - 1.0e-7;
  bool remains_active = true;
  if (grid.trim_narrow_band) {
    const bool inside = result < 0.0;
    if (inside && !(result > -grid.interior_width)) {
      result = -grid.interior_width;
      remains_active = false;
    } else if (!inside && !(result < grid.exterior_width)) {
      result = grid.exterior_width;
      remains_active = false;
    }
  }
  output[cell] = result;
  output_active[cell] = remains_active ? 1 : 0;
}

struct Runtime {
  std::mutex mutex;
  cudaStream_t stream = nullptr;
  DeviceBuffer grids;
  DeviceBuffer leaf_grid_indices;
  DeviceBuffer active;
  DeviceBuffer values;
  DeviceBuffer output;
  DeviceBuffer output_active;
  DeviceBuffer neighbour_indices;
  DeviceBuffer neighbour_values;
  PinnedBuffer host_grids;
  PinnedBuffer host_leaf_grid_indices;
  PinnedBuffer host_active;
  PinnedBuffer host_values;
  PinnedBuffer host_neighbour_indices;
  PinnedBuffer host_neighbour_values;

  ~Runtime() {
    if (stream)
      cudaStreamSynchronize(stream);
    if (stream)
      cudaStreamDestroy(stream);
  }

  void ensure_stream() {
    if (!stream) {
      cuda_memory::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create sparse renormalization CUDA stream");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }
};

ResettableRuntime<Runtime> &runtime_storage() {
  static ResettableRuntime<Runtime> state;
  return state;
}

Runtime &runtime() {
  return runtime_storage().get();
}

} // namespace

void release_renormalize_cuda_runtime() { runtime_storage().reset(); }

void renormalize_sparse_cuda(SparseRenormalizeGrid &grid) {
  std::vector<SparseRenormalizeGrid *> grids{&grid};
  renormalize_sparse_cuda_batch(grids);
}

void renormalize_sparse_cuda_batch(
    const std::vector<SparseRenormalizeGrid *> &grids) {
  size_t leaf_count = 0;
  for (SparseRenormalizeGrid *grid : grids) {
    if (!grid)
      throw std::invalid_argument(
          "sparse renormalization CUDA batch contains a null grid");
    if (!std::isfinite(grid->voxel_size) || grid->voxel_size <= 0.0)
      throw std::invalid_argument(
          "sparse renormalization voxel size must be finite and positive");
    if (!std::isfinite(grid->exterior_width) ||
        grid->exterior_width < 0.0 ||
        !std::isfinite(grid->interior_width) ||
        grid->interior_width < 0.0)
      throw std::invalid_argument(
          "sparse renormalization band widths are invalid");
    if (grid->active.size() != grid->values.size() ||
        grid->active.size() % kLeafSize != 0) {
      throw std::invalid_argument(
          "sparse renormalization leaf values are malformed");
    }
    const size_t grid_leaves = grid->active.size() / kLeafSize;
    if (grid->neighbour_indices.size() !=
            grid_leaves * kNeighbourCount ||
        grid->neighbour_values.size() !=
            grid_leaves * kNeighbourCount) {
      throw std::invalid_argument(
          "sparse renormalization neighbours are malformed");
    }
    if (grid_leaves >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        leaf_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                grid_leaves) {
      throw std::overflow_error(
          "sparse renormalization packed leaf overflow");
    }
    for (int neighbour : grid->neighbour_indices) {
      if (neighbour < -1 ||
          (neighbour >= 0 &&
           static_cast<size_t>(neighbour) >= grid_leaves)) {
        throw std::invalid_argument(
            "sparse renormalization neighbour index is invalid");
      }
    }
    leaf_count += grid_leaves;
  }
  if (grids.empty() || leaf_count == 0)
    return;
  if (leaf_count > std::numeric_limits<size_t>::max() / kLeafSize)
    throw std::overflow_error(
        "sparse renormalization packed cell overflow");
  const size_t cell_count = leaf_count * kLeafSize;
  constexpr size_t threads = 128;
  if ((cell_count + threads - 1) / threads >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "sparse renormalization CUDA launch overflow");
  }

  Runtime &state = runtime();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_stream();
  state.grids.ensure(grids.size() * sizeof(PackedRenormalizeGrid),
                     "allocate sparse renormalization grids");
  state.leaf_grid_indices.ensure(
      leaf_count * sizeof(int),
      "allocate sparse renormalization leaf owners");
  state.active.ensure(cell_count * sizeof(unsigned char),
                      "allocate sparse renormalization activity");
  state.values.ensure(cell_count * sizeof(double),
                      "allocate sparse renormalization values");
  state.output.ensure(cell_count * sizeof(double),
                      "allocate sparse renormalization output");
  state.output_active.ensure(
      cell_count * sizeof(unsigned char),
      "allocate sparse renormalization output activity");
  state.neighbour_indices.ensure(
      leaf_count * kNeighbourCount * sizeof(int),
      "allocate sparse renormalization neighbours");
  state.neighbour_values.ensure(
      leaf_count * kNeighbourCount * sizeof(double),
      "allocate sparse renormalization neighbour values");
  state.host_grids.ensure(
      grids.size() * sizeof(PackedRenormalizeGrid),
      "allocate host sparse renormalization grids");
  state.host_leaf_grid_indices.ensure(
      leaf_count * sizeof(int),
      "allocate host sparse renormalization leaf owners");
  state.host_active.ensure(
      cell_count * sizeof(unsigned char),
      "allocate host sparse renormalization activity");
  state.host_values.ensure(
      cell_count * sizeof(double),
      "allocate host sparse renormalization values");
  state.host_neighbour_indices.ensure(
      leaf_count * kNeighbourCount * sizeof(int),
      "allocate host sparse renormalization neighbours");
  state.host_neighbour_values.ensure(
      leaf_count * kNeighbourCount * sizeof(double),
      "allocate host sparse renormalization neighbour values");

  auto *host_grids =
      state.host_grids.as<PackedRenormalizeGrid>();
  int *host_owners = state.host_leaf_grid_indices.as<int>();
  unsigned char *host_active =
      state.host_active.as<unsigned char>();
  double *host_values = state.host_values.as<double>();
  int *host_neighbours =
      state.host_neighbour_indices.as<int>();
  double *host_neighbour_values =
      state.host_neighbour_values.as<double>();
  size_t leaf_offset = 0;
  for (size_t grid_index = 0; grid_index < grids.size();
       ++grid_index) {
    SparseRenormalizeGrid &grid = *grids[grid_index];
    const size_t grid_leaves = grid.active.size() / kLeafSize;
    host_grids[grid_index] =
        {leaf_offset, grid_leaves, grid.voxel_size,
         grid.exterior_width, grid.interior_width,
         grid.trim_narrow_band ? 1 : 0};
    std::fill_n(host_owners + leaf_offset, grid_leaves,
                static_cast<int>(grid_index));
    std::copy(grid.active.begin(), grid.active.end(),
              host_active + leaf_offset * kLeafSize);
    std::copy(grid.values.begin(), grid.values.end(),
              host_values + leaf_offset * kLeafSize);
    for (size_t index = 0;
         index < grid_leaves * kNeighbourCount; ++index) {
      const int local = grid.neighbour_indices[index];
      host_neighbours[leaf_offset * kNeighbourCount + index] =
          local < 0 ? -1
                    : static_cast<int>(leaf_offset) + local;
      host_neighbour_values[
          leaf_offset * kNeighbourCount + index] =
          grid.neighbour_values[index];
    }
    leaf_offset += grid_leaves;
  }

  const auto copy_to_device = [&](void *destination,
                                  const void *source, size_t bytes,
                                  const char *message) {
    cuda_memory::check(
        cudaMemcpyAsync(destination, source, bytes,
                        cudaMemcpyHostToDevice, state.stream),
        message);
  };
  copy_to_device(state.grids.as<PackedRenormalizeGrid>(),
                 host_grids,
                 grids.size() * sizeof(PackedRenormalizeGrid),
                 "copy sparse renormalization grids");
  copy_to_device(state.leaf_grid_indices.as<int>(), host_owners,
                 leaf_count * sizeof(int),
                 "copy sparse renormalization leaf owners");
  copy_to_device(state.active.as<unsigned char>(), host_active,
                 cell_count * sizeof(unsigned char),
                 "copy sparse renormalization activity");
  copy_to_device(state.values.as<double>(), host_values,
                 cell_count * sizeof(double),
                 "copy sparse renormalization values");
  copy_to_device(state.neighbour_indices.as<int>(), host_neighbours,
                 leaf_count * kNeighbourCount * sizeof(int),
                 "copy sparse renormalization neighbours");
  copy_to_device(state.neighbour_values.as<double>(),
                 host_neighbour_values,
                 leaf_count * kNeighbourCount * sizeof(double),
                 "copy sparse renormalization neighbour values");

  const int blocks =
      static_cast<int>((cell_count + threads - 1) / threads);
  renormalize_sparse_kernel<<<blocks, threads, 0, state.stream>>>(
      state.grids.as<PackedRenormalizeGrid>(),
      state.leaf_grid_indices.as<int>(), cell_count,
      state.neighbour_indices.as<int>(),
      state.neighbour_values.as<double>(),
      state.active.as<unsigned char>(), state.values.as<double>(),
      state.output.as<double>(),
      state.output_active.as<unsigned char>());
  cuda_memory::check(cudaGetLastError(),
                     "launch sparse renormalization");
  cuda_memory::check(
      cudaMemcpyAsync(host_values, state.output.as<double>(),
                      cell_count * sizeof(double),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy sparse renormalization values");
  cuda_memory::check(
      cudaMemcpyAsync(host_active,
                      state.output_active.as<unsigned char>(),
                      cell_count * sizeof(unsigned char),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy sparse renormalization activity");
  cuda_memory::check(cudaStreamSynchronize(state.stream),
                     "wait for sparse renormalization");

  leaf_offset = 0;
  for (SparseRenormalizeGrid *grid : grids) {
    std::copy_n(host_values + leaf_offset * kLeafSize,
                grid->values.size(), grid->values.begin());
    std::copy_n(host_active + leaf_offset * kLeafSize,
                grid->active.size(), grid->active.begin());
    leaf_offset += grid->active.size() / kLeafSize;
  }
}

} // namespace neural_acd
