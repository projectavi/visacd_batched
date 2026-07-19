#include <cub/device/device_scan.cuh>
#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <device_mesh.hpp>
#include <preprocess_mesh_cuda.hpp>
#include <preprocess_mesh_tables.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace neural_acd {
namespace {

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

constexpr int kSigns = 0xff;
constexpr int kInside = 0x100;
constexpr int kXEdge = 0x200;
constexpr int kYEdge = 0x400;
constexpr int kZEdge = 0x800;

__constant__ unsigned char kEdgeGroups[256 * 13];
__constant__ unsigned char kAmbiguousFaces[256];

struct PackedGrid {
  int3 minimum;
  int3 dimensions;
  int3 leaf_dimensions;
  double isovalue;
};

__device__ bool dense_offset(const PackedGrid &grid, int3 coordinate,
                             size_t &offset) {
  const int x = coordinate.x - grid.minimum.x;
  const int y = coordinate.y - grid.minimum.y;
  const int z = coordinate.z - grid.minimum.z;
  if (x < 0 || y < 0 || z < 0 || x >= grid.dimensions.x ||
      y >= grid.dimensions.y || z >= grid.dimensions.z)
    return false;
  offset = (static_cast<size_t>(x) * grid.dimensions.y + y) *
               grid.dimensions.z +
           z;
  return true;
}

__device__ int3 dense_coordinate(const PackedGrid &grid, size_t offset) {
  const size_t yz = static_cast<size_t>(grid.dimensions.y) *
                    grid.dimensions.z;
  const int x = static_cast<int>(offset / yz);
  offset %= yz;
  const int y = static_cast<int>(offset / grid.dimensions.z);
  const int z = static_cast<int>(offset % grid.dimensions.z);
  return make_int3(grid.minimum.x + x, grid.minimum.y + y,
                   grid.minimum.z + z);
}

__device__ int3 ordered_coordinate(const PackedGrid &grid,
                                   size_t ordered,
                                   const int *leaf_order) {
  const size_t local = ordered & 511;
  const size_t leaf =
      static_cast<size_t>(leaf_order[ordered >> 9]);
  const size_t leaf_yz =
      static_cast<size_t>(grid.leaf_dimensions.y) *
      grid.leaf_dimensions.z;
  const int leaf_x = static_cast<int>(leaf / leaf_yz);
  const size_t leaf_remainder = leaf % leaf_yz;
  const int leaf_y =
      static_cast<int>(leaf_remainder / grid.leaf_dimensions.z);
  const int leaf_z =
      static_cast<int>(leaf_remainder % grid.leaf_dimensions.z);
  const int x = static_cast<int>(local / 64);
  const int y = static_cast<int>((local / 8) & 7);
  const int z = static_cast<int>(local & 7);
  return make_int3(grid.minimum.x + leaf_x * 8 + x,
                   grid.minimum.y + leaf_y * 8 + y,
                   grid.minimum.z + leaf_z * 8 + z);
}

__device__ bool ordered_offset(const PackedGrid &grid, int3 coordinate,
                               const int *leaf_ranks,
                               size_t &ordered) {
  const int x = coordinate.x - grid.minimum.x;
  const int y = coordinate.y - grid.minimum.y;
  const int z = coordinate.z - grid.minimum.z;
  if (x < 0 || y < 0 || z < 0 || x >= grid.dimensions.x ||
      y >= grid.dimensions.y || z >= grid.dimensions.z)
    return false;
  const size_t leaf =
      (static_cast<size_t>(x >> 3) * grid.leaf_dimensions.y +
       (y >> 3)) *
          grid.leaf_dimensions.z +
      (z >> 3);
  ordered = static_cast<size_t>(leaf_ranks[leaf]) * 512 +
            (static_cast<size_t>(x & 7) * 8 + (y & 7)) * 8 +
            (z & 7);
  return true;
}

__device__ unsigned char cell_signs(const PackedGrid &grid,
                                    int3 coordinate,
                                    const double *values) {
  const int3 corners[8] = {
      make_int3(0, 0, 0), make_int3(1, 0, 0),
      make_int3(1, 0, 1), make_int3(0, 0, 1),
      make_int3(0, 1, 0), make_int3(1, 1, 0),
      make_int3(1, 1, 1), make_int3(0, 1, 1)};
  unsigned int signs = 0;
  for (int corner = 0; corner < 8; ++corner) {
    size_t offset = 0;
    const int3 point = make_int3(coordinate.x + corners[corner].x,
                                 coordinate.y + corners[corner].y,
                                 coordinate.z + corners[corner].z);
    if (dense_offset(grid, point, offset) &&
        values[offset] < grid.isovalue)
      signs |= 1u << corner;
  }
  return static_cast<unsigned char>(signs);
}

__device__ void mark_intersection_cell(const PackedGrid &grid,
                                       int3 coordinate,
                                       int *intersection) {
  size_t offset = 0;
  if (dense_offset(grid, coordinate, offset))
    atomicExch(intersection + offset, 1);
}

__global__ void identify_intersections_kernel(
    PackedGrid grid, size_t cell_count, const unsigned char *active,
    const double *values, int *intersection) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count)
    return;
  const int3 coordinate = dense_coordinate(grid, cell);
  const bool inside = values[cell] < grid.isovalue;
  for (int axis = 0; axis < 3; ++axis) {
    int3 adjacent = coordinate;
    if (axis == 0)
      ++adjacent.x;
    else if (axis == 1)
      ++adjacent.y;
    else
      ++adjacent.z;
    size_t adjacent_offset = 0;
    if (!dense_offset(grid, adjacent, adjacent_offset) ||
        (!active[cell] && !active[adjacent_offset]) ||
        inside == (values[adjacent_offset] < grid.isovalue))
      continue;
    mark_intersection_cell(grid, coordinate, intersection);
    if (axis == 0) {
      mark_intersection_cell(
          grid, make_int3(coordinate.x, coordinate.y - 1,
                          coordinate.z),
          intersection);
      mark_intersection_cell(
          grid, make_int3(coordinate.x, coordinate.y - 1,
                          coordinate.z - 1),
          intersection);
      mark_intersection_cell(
          grid, make_int3(coordinate.x, coordinate.y,
                          coordinate.z - 1),
          intersection);
    } else if (axis == 1) {
      mark_intersection_cell(
          grid, make_int3(coordinate.x, coordinate.y,
                          coordinate.z - 1),
          intersection);
      mark_intersection_cell(
          grid, make_int3(coordinate.x - 1, coordinate.y,
                          coordinate.z - 1),
          intersection);
      mark_intersection_cell(
          grid, make_int3(coordinate.x - 1, coordinate.y,
                          coordinate.z),
          intersection);
    } else {
      mark_intersection_cell(
          grid, make_int3(coordinate.x, coordinate.y - 1,
                          coordinate.z),
          intersection);
      mark_intersection_cell(
          grid, make_int3(coordinate.x - 1, coordinate.y - 1,
                          coordinate.z),
          intersection);
      mark_intersection_cell(
          grid, make_int3(coordinate.x - 1, coordinate.y,
                          coordinate.z),
          intersection);
    }
  }
}

__device__ unsigned char correct_ambiguous_signs(
    PackedGrid grid, int3 coordinate, unsigned char signs,
    const double *values) {
  const unsigned char face = kAmbiguousFaces[signs];
  int3 adjacent = coordinate;
  unsigned char opposite = 0;
  if (face == 1) {
    --adjacent.z;
    opposite = 3;
  } else if (face == 2) {
    ++adjacent.x;
    opposite = 4;
  } else if (face == 3) {
    ++adjacent.z;
    opposite = 1;
  } else if (face == 4) {
    --adjacent.x;
    opposite = 2;
  } else if (face == 5) {
    --adjacent.y;
    opposite = 6;
  } else if (face == 6) {
    ++adjacent.y;
    opposite = 5;
  }
  if (opposite != 0 &&
      kAmbiguousFaces[cell_signs(grid, adjacent, values)] == opposite)
    signs = static_cast<unsigned char>(~signs);
  return signs;
}

__global__ void compute_surface_flags_kernel(
    PackedGrid grid, size_t cell_count, const int *intersection,
    const double *values, const int *leaf_order,
    unsigned short *flags,
    unsigned int *point_counts, unsigned int *quad_counts) {
  const size_t ordered =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ordered >= cell_count)
    return;
  const int3 coordinate =
      ordered_coordinate(grid, ordered, leaf_order);
  size_t dense = 0;
  if (!dense_offset(grid, coordinate, dense) || !intersection[dense]) {
    flags[ordered] = 0;
    point_counts[ordered] = 0;
    quad_counts[ordered] = 0;
    return;
  }
  const unsigned char raw = cell_signs(grid, coordinate, values);
  if (raw == 0 || raw == 0xff) {
    flags[ordered] = 0;
    point_counts[ordered] = 0;
    quad_counts[ordered] = 0;
    return;
  }
  const bool inside = (raw & 1) != 0;
  int edge_flags = inside ? kInside : 0;
  if (inside != ((raw & 0x02) != 0))
    edge_flags |= kXEdge;
  if (inside != ((raw & 0x10) != 0))
    edge_flags |= kYEdge;
  if (inside != ((raw & 0x08) != 0))
    edge_flags |= kZEdge;
  const unsigned char corrected =
      correct_ambiguous_signs(grid, coordinate, raw, values);
  flags[ordered] =
      static_cast<unsigned short>(edge_flags | corrected);
  point_counts[ordered] = kEdgeGroups[corrected * 13];
  quad_counts[ordered] =
      ((edge_flags & kXEdge) ? 1u : 0u) +
      ((edge_flags & kYEdge) ? 1u : 0u) +
      ((edge_flags & kZEdge) ? 1u : 0u);
}

__device__ double zero_crossing(double first, double second,
                                double isovalue) {
  return (isovalue - first) / (second - first);
}

__device__ double3 compute_point(const unsigned char *groups,
                                 const double *values,
                                 unsigned char edge_group,
                                 double isovalue) {
  double3 point = make_double3(0.0, 0.0, 0.0);
  int samples = 0;
  if (groups[1] == edge_group) {
    point.x += zero_crossing(values[0], values[1], isovalue);
    ++samples;
  }
  if (groups[2] == edge_group) {
    point.x += 1.0;
    point.z += zero_crossing(values[1], values[2], isovalue);
    ++samples;
  }
  if (groups[3] == edge_group) {
    point.x += zero_crossing(values[3], values[2], isovalue);
    point.z += 1.0;
    ++samples;
  }
  if (groups[4] == edge_group) {
    point.z += zero_crossing(values[0], values[3], isovalue);
    ++samples;
  }
  if (groups[5] == edge_group) {
    point.x += zero_crossing(values[4], values[5], isovalue);
    point.y += 1.0;
    ++samples;
  }
  if (groups[6] == edge_group) {
    point.x += 1.0;
    point.y += 1.0;
    point.z += zero_crossing(values[5], values[6], isovalue);
    ++samples;
  }
  if (groups[7] == edge_group) {
    point.x += zero_crossing(values[7], values[6], isovalue);
    point.y += 1.0;
    point.z += 1.0;
    ++samples;
  }
  if (groups[8] == edge_group) {
    point.y += 1.0;
    point.z += zero_crossing(values[4], values[7], isovalue);
    ++samples;
  }
  if (groups[9] == edge_group) {
    point.y += zero_crossing(values[0], values[4], isovalue);
    ++samples;
  }
  if (groups[10] == edge_group) {
    point.x += 1.0;
    point.y += zero_crossing(values[1], values[5], isovalue);
    ++samples;
  }
  if (groups[11] == edge_group) {
    point.x += 1.0;
    point.y += zero_crossing(values[2], values[6], isovalue);
    point.z += 1.0;
    ++samples;
  }
  if (groups[12] == edge_group) {
    point.y += zero_crossing(values[3], values[7], isovalue);
    point.z += 1.0;
    ++samples;
  }
  if (samples > 1) {
    const double weight = 1.0 / static_cast<double>(samples);
    point.x *= weight;
    point.y *= weight;
    point.z *= weight;
  }
  return point;
}

__global__ void compute_points_kernel(
    PackedGrid grid, size_t cell_count, const double *grid_values,
    const int *leaf_order, const unsigned short *flags,
    const unsigned int *point_offsets,
    float3 *points) {
  const size_t ordered =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ordered >= cell_count || !flags[ordered])
    return;
  const int3 coordinate =
      ordered_coordinate(grid, ordered, leaf_order);
  const int3 corners[8] = {
      make_int3(0, 0, 0), make_int3(1, 0, 0),
      make_int3(1, 0, 1), make_int3(0, 0, 1),
      make_int3(0, 1, 0), make_int3(1, 1, 0),
      make_int3(1, 1, 1), make_int3(0, 1, 1)};
  double values[8];
  for (int corner = 0; corner < 8; ++corner) {
    size_t offset = 0;
    dense_offset(grid,
                 make_int3(coordinate.x + corners[corner].x,
                           coordinate.y + corners[corner].y,
                           coordinate.z + corners[corner].z),
                 offset);
    values[corner] = grid_values[offset];
  }
  const unsigned char signs =
      static_cast<unsigned char>(flags[ordered] & kSigns);
  const unsigned char *groups = kEdgeGroups + signs * 13;
  const unsigned int output = point_offsets[ordered];
  for (unsigned char group = 1; group <= groups[0]; ++group) {
    double3 point = compute_point(groups, values, group, grid.isovalue);
    point.x += coordinate.x;
    point.y += coordinate.y;
    point.z += coordinate.z;
    points[output + group - 1] =
        make_float3(static_cast<float>(point.x),
                    static_cast<float>(point.y),
                    static_cast<float>(point.z));
  }
}

__device__ bool point_index(const PackedGrid &grid, int3 coordinate,
                            const int *leaf_ranks,
                            const unsigned short *flags,
                            const unsigned int *point_offsets,
                            unsigned int &index, int edge_slot) {
  size_t ordered = 0;
  if (!ordered_offset(grid, coordinate, leaf_ranks, ordered) ||
      !flags[ordered])
    return false;
  const unsigned char signs =
      static_cast<unsigned char>(flags[ordered] & kSigns);
  index = point_offsets[ordered];
  const unsigned char count = kEdgeGroups[signs * 13];
  if (count > 1)
    index += kEdgeGroups[signs * 13 + edge_slot] - 1;
  return true;
}

__device__ int4 orient_quad(unsigned int first, unsigned int second,
                            unsigned int third, unsigned int fourth,
                            bool reverse) {
  if (reverse)
    return make_int4(static_cast<int>(fourth),
                     static_cast<int>(third),
                     static_cast<int>(second),
                     static_cast<int>(first));
  return make_int4(static_cast<int>(first),
                   static_cast<int>(second),
                   static_cast<int>(third),
                   static_cast<int>(fourth));
}

__global__ void compute_quads_kernel(
    PackedGrid grid, size_t cell_count, const int *leaf_order,
    const int *leaf_ranks, const unsigned short *flags,
    const unsigned int *point_offsets,
    const unsigned int *quad_offsets, int4 *quads) {
  const size_t ordered =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ordered >= cell_count)
    return;
  const unsigned short cell_flags = flags[ordered];
  if (!(cell_flags & (kXEdge | kYEdge | kZEdge)))
    return;
  const int3 coordinate =
      ordered_coordinate(grid, ordered, leaf_order);
  const unsigned char signs =
      static_cast<unsigned char>(cell_flags & kSigns);
  const unsigned char *groups = kEdgeGroups + signs * 13;
  unsigned int output = quad_offsets[ordered];
  unsigned int first, second, third, fourth;
  const bool inside = (cell_flags & kInside) != 0;
  if (cell_flags & kXEdge) {
    first = point_offsets[ordered] +
            (groups[0] > 1 ? groups[1] - 1 : 0);
    bool valid = point_index(
        grid, make_int3(coordinate.x, coordinate.y - 1,
                        coordinate.z),
        leaf_ranks, flags, point_offsets, second, 5);
    valid &= point_index(
        grid, make_int3(coordinate.x, coordinate.y - 1,
                        coordinate.z - 1),
        leaf_ranks, flags, point_offsets, third, 7);
    valid &= point_index(
        grid, make_int3(coordinate.x, coordinate.y,
                        coordinate.z - 1),
        leaf_ranks, flags, point_offsets, fourth, 3);
    if (valid)
      quads[output++] = orient_quad(first, second, third, fourth,
                                    inside);
  }
  if (cell_flags & kYEdge) {
    first = point_offsets[ordered] +
            (groups[0] > 1 ? groups[9] - 1 : 0);
    bool valid = point_index(
        grid, make_int3(coordinate.x, coordinate.y,
                        coordinate.z - 1),
        leaf_ranks, flags, point_offsets, second, 12);
    valid &= point_index(
        grid, make_int3(coordinate.x - 1, coordinate.y,
                        coordinate.z - 1),
        leaf_ranks, flags, point_offsets, third, 11);
    valid &= point_index(
        grid, make_int3(coordinate.x - 1, coordinate.y,
                        coordinate.z),
        leaf_ranks, flags, point_offsets, fourth, 10);
    if (valid)
      quads[output++] = orient_quad(first, second, third, fourth,
                                    inside);
  }
  if (cell_flags & kZEdge) {
    first = point_offsets[ordered] +
            (groups[0] > 1 ? groups[4] - 1 : 0);
    bool valid = point_index(
        grid, make_int3(coordinate.x, coordinate.y - 1,
                        coordinate.z),
        leaf_ranks, flags, point_offsets, second, 8);
    valid &= point_index(
        grid, make_int3(coordinate.x - 1, coordinate.y - 1,
                        coordinate.z),
        leaf_ranks, flags, point_offsets, third, 6);
    valid &= point_index(
        grid, make_int3(coordinate.x - 1, coordinate.y,
                        coordinate.z),
        leaf_ranks, flags, point_offsets, fourth, 2);
    if (valid)
      quads[output++] = orient_quad(first, second, third, fourth,
                                    !inside);
  }
}

struct Runtime {
  std::mutex mutex;
  cudaStream_t stream = nullptr;
  cudaEvent_t input_ready = nullptr;
  bool tables_ready = false;
  DeviceBuffer active, values, intersection, flags;
  DeviceBuffer leaf_order, leaf_ranks;
  DeviceBuffer point_counts, point_offsets;
  DeviceBuffer quad_counts, quad_offsets;
  DeviceBuffer scan_storage, points, quads;
  PinnedBuffer host_active, host_values, host_points, host_quads;
  PinnedBuffer host_leaf_order, host_leaf_ranks;
  PinnedBuffer host_totals;

  ~Runtime() {
    if (input_ready)
      cudaEventDestroy(input_ready);
    if (stream)
      cudaStreamDestroy(stream);
  }

  void ensure_stream() {
    if (!stream) {
      cuda_memory::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create dense volume meshing CUDA stream");
    }
    if (!input_ready) {
      cuda_memory::check(
          cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming),
          "create dense volume meshing input event");
    }
    DeviceBuffer::set_allocation_stream(stream);
    if (!tables_ready) {
      unsigned char edge_groups[256 * 13];
      unsigned char ambiguous_faces[256];
      copy_openvdb_volume_mesh_tables(edge_groups, ambiguous_faces);
      cuda_memory::check(
          cudaMemcpyToSymbolAsync(kEdgeGroups, edge_groups,
                                  sizeof(edge_groups), 0,
                                  cudaMemcpyHostToDevice, stream),
          "copy OpenVDB edge group table");
      cuda_memory::check(
          cudaMemcpyToSymbolAsync(kAmbiguousFaces, ambiguous_faces,
                                  sizeof(ambiguous_faces), 0,
                                  cudaMemcpyHostToDevice, stream),
          "copy OpenVDB ambiguous face table");
      tables_ready = true;
    }
  }
};

Runtime &runtime() {
  static Runtime state;
  return state;
}

} // namespace

void mesh_dense_volume_cuda(DenseVolumeMeshingGrid &grid) {
  std::vector<DenseVolumeMeshingGrid *> grids{&grid};
  mesh_dense_volume_cuda_batch(grids);
}

namespace {

void mesh_dense_volume_cuda_batch_impl(
    const std::vector<DenseVolumeMeshingGrid *> &grids,
    const unsigned char *device_active, const double *device_values,
    const std::vector<size_t> *source_cell_offsets,
    cudaStream_t producer_stream) {
  const bool device_input = device_active != nullptr;
  if (device_input != (device_values != nullptr) ||
      device_input != (source_cell_offsets != nullptr) ||
      device_input != (producer_stream != nullptr) ||
      (source_cell_offsets && source_cell_offsets->size() != grids.size()))
    throw std::invalid_argument(
        "dense volume meshing device input is malformed");
  struct BatchGrid {
    PackedGrid packed;
    size_t cell_offset = 0;
    size_t cell_count = 0;
    size_t leaf_offset = 0;
    size_t leaf_count = 0;
    size_t point_count = 0;
    size_t quad_count = 0;
    size_t point_output_offset = 0;
    size_t quad_output_offset = 0;
  };
  std::vector<BatchGrid> batch(grids.size());
  size_t cell_count = 0, leaf_count = 0;
  for (size_t grid_index = 0; grid_index < grids.size(); ++grid_index) {
    DenseVolumeMeshingGrid *grid = grids[grid_index];
    if (!grid || (grid->minimum[0] & 7) != 0 ||
        (grid->minimum[1] & 7) != 0 ||
        (grid->minimum[2] & 7) != 0 || grid->dimensions[0] <= 0 ||
        grid->dimensions[1] <= 0 || grid->dimensions[2] <= 0 ||
        (grid->dimensions[0] & 7) != 0 ||
        (grid->dimensions[1] & 7) != 0 ||
        (grid->dimensions[2] & 7) != 0 ||
        !std::isfinite(grid->isovalue))
      throw std::invalid_argument("dense volume meshing grid is invalid");
    size_t grid_cells = static_cast<size_t>(grid->dimensions[0]);
    if (grid_cells > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(grid->dimensions[1]))
      throw std::overflow_error("dense volume meshing cell overflow");
    grid_cells *= static_cast<size_t>(grid->dimensions[1]);
    if (grid_cells > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(grid->dimensions[2]))
      throw std::overflow_error("dense volume meshing cell overflow");
    grid_cells *= static_cast<size_t>(grid->dimensions[2]);
    if ((!device_input &&
         (grid->active.size() != grid_cells ||
          grid->values.size() != grid_cells)) ||
        grid_cells == 0 ||
        grid_cells >
            static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument(
          "dense volume meshing arrays are malformed");
    const size_t grid_leaves = grid_cells / 512;
    if (grid->leaf_order.size() != grid_leaves)
      throw std::invalid_argument(
          "dense volume meshing leaf order is malformed");
    std::vector<unsigned char> seen(grid_leaves, 0);
    for (int leaf : grid->leaf_order) {
      if (leaf < 0 || static_cast<size_t>(leaf) >= grid_leaves ||
          seen[leaf])
        throw std::invalid_argument(
            "dense volume meshing leaf order is not a permutation");
      seen[leaf] = 1;
    }
    if (cell_count > std::numeric_limits<size_t>::max() - grid_cells ||
        leaf_count > std::numeric_limits<size_t>::max() - grid_leaves)
      throw std::overflow_error("dense volume meshing batch overflow");
    BatchGrid &entry = batch[grid_index];
    entry.packed = {
        make_int3(grid->minimum[0], grid->minimum[1],
                  grid->minimum[2]),
        make_int3(grid->dimensions[0], grid->dimensions[1],
                  grid->dimensions[2]),
        make_int3(grid->dimensions[0] / 8,
                  grid->dimensions[1] / 8,
                  grid->dimensions[2] / 8),
        grid->isovalue};
    entry.cell_offset = cell_count;
    entry.cell_count = grid_cells;
    entry.leaf_offset = leaf_count;
    entry.leaf_count = grid_leaves;
    cell_count += grid_cells;
    leaf_count += grid_leaves;
  }
  if (grids.empty())
    return;

  Runtime &state = runtime();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_stream();
#define ENSURE(buffer, bytes, message) state.buffer.ensure(bytes, message)
  if (!device_input) {
    ENSURE(active, cell_count * sizeof(unsigned char),
           "allocate batched dense volume activity");
    ENSURE(values, cell_count * sizeof(double),
           "allocate batched dense volume values");
  }
  ENSURE(intersection, cell_count * sizeof(int),
         "allocate batched dense volume intersections");
  ENSURE(flags, cell_count * sizeof(unsigned short),
         "allocate batched dense volume flags");
  ENSURE(leaf_order, leaf_count * sizeof(int),
         "allocate batched dense volume leaf order");
  ENSURE(leaf_ranks, leaf_count * sizeof(int),
         "allocate batched dense volume leaf ranks");
  ENSURE(point_counts, cell_count * sizeof(unsigned int),
         "allocate batched dense volume point counts");
  ENSURE(point_offsets, cell_count * sizeof(unsigned int),
         "allocate batched dense volume point offsets");
  ENSURE(quad_counts, cell_count * sizeof(unsigned int),
         "allocate batched dense volume quad counts");
  ENSURE(quad_offsets, cell_count * sizeof(unsigned int),
         "allocate batched dense volume quad offsets");
  if (!device_input) {
    ENSURE(host_active, cell_count * sizeof(unsigned char),
           "allocate host batched dense volume activity");
    ENSURE(host_values, cell_count * sizeof(double),
           "allocate host batched dense volume values");
  }
  ENSURE(host_leaf_order, leaf_count * sizeof(int),
         "allocate host batched dense volume leaf order");
  ENSURE(host_leaf_ranks, leaf_count * sizeof(int),
         "allocate host batched dense volume leaf ranks");
  ENSURE(host_totals, grids.size() * 4 * sizeof(unsigned int),
         "allocate host batched dense volume totals");
#undef ENSURE

  for (size_t grid_index = 0; grid_index < grids.size(); ++grid_index) {
    const DenseVolumeMeshingGrid &grid = *grids[grid_index];
    const BatchGrid &entry = batch[grid_index];
    if (!device_input) {
      std::copy(grid.active.begin(), grid.active.end(),
                state.host_active.as<unsigned char>() +
                    entry.cell_offset);
      std::copy(grid.values.begin(), grid.values.end(),
                state.host_values.as<double>() + entry.cell_offset);
    }
    for (size_t rank = 0; rank < entry.leaf_count; ++rank) {
      const int local = grid.leaf_order[rank];
      state.host_leaf_order.as<int>()[entry.leaf_offset + rank] =
          local;
      state.host_leaf_ranks.as<int>()[entry.leaf_offset + local] =
          static_cast<int>(rank);
    }
  }
  const auto copy_to_device = [&](void *destination, const void *source,
                                  size_t bytes, const char *message) {
    cuda_memory::check(
        cudaMemcpyAsync(destination, source, bytes,
                        cudaMemcpyHostToDevice, state.stream),
        message);
  };
  if (device_input) {
    cuda_memory::check(cudaEventRecord(state.input_ready, producer_stream),
                       "record dense volume device input event");
    cuda_memory::check(
        cudaStreamWaitEvent(state.stream, state.input_ready, 0),
        "wait for dense volume device input");
  } else {
    copy_to_device(state.active.as<unsigned char>(),
                   state.host_active.as<unsigned char>(),
                   cell_count * sizeof(unsigned char),
                   "copy batched dense volume activity");
    copy_to_device(state.values.as<double>(),
                   state.host_values.as<double>(),
                   cell_count * sizeof(double),
                   "copy batched dense volume values");
  }
  copy_to_device(state.leaf_order.as<int>(),
                 state.host_leaf_order.as<int>(),
                 leaf_count * sizeof(int),
                 "copy batched dense volume leaf order");
  copy_to_device(state.leaf_ranks.as<int>(),
                 state.host_leaf_ranks.as<int>(),
                 leaf_count * sizeof(int),
                 "copy batched dense volume leaf ranks");
  cuda_memory::check(
      cudaMemsetAsync(state.intersection.as<int>(), 0,
                      cell_count * sizeof(int), state.stream),
      "clear batched dense volume intersections");

  size_t scan_bytes = 0;
  for (const BatchGrid &entry : batch) {
    size_t bytes = 0;
    cuda_memory::check(
        cub::DeviceScan::ExclusiveSum(
            nullptr, bytes,
            state.point_counts.as<unsigned int>() + entry.cell_offset,
            state.point_offsets.as<unsigned int>() + entry.cell_offset,
            static_cast<int>(entry.cell_count), state.stream),
        "size batched dense volume scan");
    scan_bytes = std::max(scan_bytes, bytes);
  }
  state.scan_storage.ensure(scan_bytes,
                            "allocate batched dense volume scan storage");
  constexpr int threads = 128;
  unsigned int *host_totals = state.host_totals.as<unsigned int>();
  for (size_t grid_index = 0; grid_index < batch.size(); ++grid_index) {
    const BatchGrid &entry = batch[grid_index];
    const int blocks = static_cast<int>(
        (entry.cell_count + threads - 1) / threads);
    const size_t input_offset =
        device_input ? (*source_cell_offsets)[grid_index]
                     : entry.cell_offset;
    const unsigned char *active =
        device_input ? device_active + input_offset
                     : state.active.as<unsigned char>() + input_offset;
    const double *values =
        device_input ? device_values + input_offset
                     : state.values.as<double>() + input_offset;
    int *intersection = state.intersection.as<int>() + entry.cell_offset;
    unsigned short *flags = state.flags.as<unsigned short>() +
                            entry.cell_offset;
    unsigned int *point_counts =
        state.point_counts.as<unsigned int>() + entry.cell_offset;
    unsigned int *point_offsets =
        state.point_offsets.as<unsigned int>() + entry.cell_offset;
    unsigned int *quad_counts =
        state.quad_counts.as<unsigned int>() + entry.cell_offset;
    unsigned int *quad_offsets =
        state.quad_offsets.as<unsigned int>() + entry.cell_offset;
    int *leaf_order = state.leaf_order.as<int>() + entry.leaf_offset;
    identify_intersections_kernel<<<blocks, threads, 0, state.stream>>>(
        entry.packed, entry.cell_count, active, values, intersection);
    cuda_memory::check(cudaGetLastError(),
                       "identify batched dense volume intersections");
    compute_surface_flags_kernel<<<blocks, threads, 0, state.stream>>>(
        entry.packed, entry.cell_count, intersection, values, leaf_order,
        flags, point_counts, quad_counts);
    cuda_memory::check(cudaGetLastError(),
                       "compute batched dense volume surface flags");
    size_t bytes = scan_bytes;
    cuda_memory::check(
        cub::DeviceScan::ExclusiveSum(
            state.scan_storage.as<void>(), bytes, point_counts,
            point_offsets, static_cast<int>(entry.cell_count),
            state.stream),
        "scan batched dense volume point counts");
    cuda_memory::check(
        cudaMemcpyAsync(host_totals + grid_index * 4,
                        point_offsets + entry.cell_count - 1,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost,
                        state.stream),
        "copy batched dense volume final point offset");
    cuda_memory::check(
        cudaMemcpyAsync(host_totals + grid_index * 4 + 1,
                        point_counts + entry.cell_count - 1,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost,
                        state.stream),
        "copy batched dense volume final point count");
    bytes = scan_bytes;
    cuda_memory::check(
        cub::DeviceScan::ExclusiveSum(
            state.scan_storage.as<void>(), bytes, quad_counts,
            quad_offsets, static_cast<int>(entry.cell_count),
            state.stream),
        "scan batched dense volume quad counts");
    cuda_memory::check(
        cudaMemcpyAsync(host_totals + grid_index * 4 + 2,
                        quad_offsets + entry.cell_count - 1,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost,
                        state.stream),
        "copy batched dense volume final quad offset");
    cuda_memory::check(
        cudaMemcpyAsync(host_totals + grid_index * 4 + 3,
                        quad_counts + entry.cell_count - 1,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost,
                        state.stream),
        "copy batched dense volume final quad count");
  }
  cuda_memory::check(cudaStreamSynchronize(state.stream),
                     "wait for batched dense volume counts");

  size_t point_count = 0, quad_count = 0;
  for (size_t grid_index = 0; grid_index < batch.size(); ++grid_index) {
    BatchGrid &entry = batch[grid_index];
    entry.point_count =
        static_cast<size_t>(host_totals[grid_index * 4]) +
        host_totals[grid_index * 4 + 1];
    entry.quad_count =
        static_cast<size_t>(host_totals[grid_index * 4 + 2]) +
        host_totals[grid_index * 4 + 3];
    if (entry.point_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        entry.quad_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        point_count > std::numeric_limits<size_t>::max() -
                          entry.point_count ||
        quad_count > std::numeric_limits<size_t>::max() -
                         entry.quad_count)
      throw std::overflow_error("batched dense volume output overflow");
    entry.point_output_offset = point_count;
    entry.quad_output_offset = quad_count;
    point_count += entry.point_count;
    quad_count += entry.quad_count;
  }
  state.points.ensure(point_count * sizeof(float3),
                      "allocate batched dense volume points");
  state.quads.ensure(quad_count * sizeof(int4),
                     "allocate batched dense volume quads");
  state.host_points.ensure(point_count * sizeof(float3),
                           "allocate host batched dense volume points");
  state.host_quads.ensure(quad_count * sizeof(int4),
                          "allocate host batched dense volume quads");
  for (size_t grid_index = 0; grid_index < batch.size(); ++grid_index) {
    const BatchGrid &entry = batch[grid_index];
    const int blocks = static_cast<int>(
        (entry.cell_count + threads - 1) / threads);
    if (entry.point_count > 0) {
      const size_t input_offset =
          device_input ? (*source_cell_offsets)[grid_index]
                       : entry.cell_offset;
      const double *values =
          device_input ? device_values + input_offset
                       : state.values.as<double>() + input_offset;
      compute_points_kernel<<<blocks, threads, 0, state.stream>>>(
          entry.packed, entry.cell_count, values,
          state.leaf_order.as<int>() + entry.leaf_offset,
          state.flags.as<unsigned short>() + entry.cell_offset,
          state.point_offsets.as<unsigned int>() + entry.cell_offset,
          state.points.as<float3>() + entry.point_output_offset);
      cuda_memory::check(cudaGetLastError(),
                         "compute batched dense volume points");
    }
    if (entry.quad_count > 0) {
      compute_quads_kernel<<<blocks, threads, 0, state.stream>>>(
          entry.packed, entry.cell_count,
          state.leaf_order.as<int>() + entry.leaf_offset,
          state.leaf_ranks.as<int>() + entry.leaf_offset,
          state.flags.as<unsigned short>() + entry.cell_offset,
          state.point_offsets.as<unsigned int>() + entry.cell_offset,
          state.quad_offsets.as<unsigned int>() + entry.cell_offset,
          state.quads.as<int4>() + entry.quad_output_offset);
      cuda_memory::check(cudaGetLastError(),
                         "compute batched dense volume quads");
    }
    DenseVolumeMeshingGrid &grid = *grids[grid_index];
    grid.device_mesh.reset();
    if (grid.retain_device_mesh && entry.point_count > 0 &&
        entry.quad_count > 0) {
      grid.device_mesh = try_make_device_mesh_from_quads(
          reinterpret_cast<const float *>(
              state.points.as<float3>() + entry.point_output_offset),
          entry.point_count,
          reinterpret_cast<const int *>(
              state.quads.as<int4>() + entry.quad_output_offset),
          entry.quad_count, grid.output_scale,
          reinterpret_cast<void *>(state.stream),
          grid.device_memory_fraction);
    }
  }
  if (point_count > 0) {
    cuda_memory::check(
        cudaMemcpyAsync(state.host_points.as<float3>(),
                        state.points.as<float3>(),
                        point_count * sizeof(float3),
                        cudaMemcpyDeviceToHost, state.stream),
        "copy batched dense volume points");
  }
  if (quad_count > 0) {
    cuda_memory::check(
        cudaMemcpyAsync(state.host_quads.as<int4>(),
                        state.quads.as<int4>(),
                        quad_count * sizeof(int4),
                        cudaMemcpyDeviceToHost, state.stream),
        "copy batched dense volume quads");
  }
  cuda_memory::check(cudaStreamSynchronize(state.stream),
                     "wait for batched dense volume meshing");
  for (size_t grid_index = 0; grid_index < grids.size(); ++grid_index) {
    DenseVolumeMeshingGrid &grid = *grids[grid_index];
    const BatchGrid &entry = batch[grid_index];
    grid.points.resize(entry.point_count * 3);
    grid.quads.resize(entry.quad_count * 4);
    if (entry.point_count > 0) {
      std::copy_n(state.host_points.as<float>() +
                      entry.point_output_offset * 3,
                  entry.point_count * 3, grid.points.begin());
    }
    if (entry.quad_count > 0) {
      std::copy_n(state.host_quads.as<int>() +
                      entry.quad_output_offset * 4,
                  entry.quad_count * 4, grid.quads.begin());
    }
  }
}

} // namespace

void mesh_dense_volume_cuda_batch(
    const std::vector<DenseVolumeMeshingGrid *> &grids) {
  mesh_dense_volume_cuda_batch_impl(grids, nullptr, nullptr, nullptr,
                                    nullptr);
}

void mesh_dense_volume_cuda_device_batch(
    const std::vector<DenseVolumeMeshingGrid *> &grids,
    const unsigned char *device_active, const double *device_values,
    const std::vector<size_t> &cell_offsets, void *producer_stream) {
  mesh_dense_volume_cuda_batch_impl(
      grids, device_active, device_values, &cell_offsets,
      reinterpret_cast<cudaStream_t>(producer_stream));
}

} // namespace neural_acd
