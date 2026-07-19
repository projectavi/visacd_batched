#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <preprocess_expand_cuda.hpp>
#include <preprocess_mesh_cuda.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace neural_acd {
namespace {

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

struct PackedNarrowbandCandidate {
  int x;
  int y;
  int z;
  int manhattan_limit;
  size_t fragment_offset;
  size_t fragment_count;
  int triangle_offset;
  double voxel_size;
};

struct PackedDenseGrid {
  int3 minimum;
  int3 dimensions;
  int3 leaf_dimensions;
  size_t cell_offset;
  size_t cell_count;
  size_t leaf_offset;
  size_t vertex_offset;
  size_t triangle_offset;
  double exterior_width;
  double interior_width;
  double voxel_size;
  unsigned int iterations;
  int renormalize;
};

__device__ double3 subtract(double3 first, double3 second) {
  return make_double3(first.x - second.x, first.y - second.y,
                      first.z - second.z);
}

__device__ double3 add_scaled(double3 point, double3 direction,
                              double scale) {
  return make_double3(point.x + direction.x * scale,
                      point.y + direction.y * scale,
                      point.z + direction.z * scale);
}

__device__ double dot_product(double3 first, double3 second) {
  return first.x * second.x + first.y * second.y +
         first.z * second.z;
}

__device__ double3 closest_triangle(double3 a, double3 b, double3 c,
                                    double3 point) {
  const double3 ab = subtract(b, a);
  const double3 ac = subtract(c, a);
  const double3 ap = subtract(point, a);
  const double d1 = dot_product(ab, ap);
  const double d2 = dot_product(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0)
    return a;

  const double3 bp = subtract(point, b);
  const double d3 = dot_product(ab, bp);
  const double d4 = dot_product(ac, bp);
  if (d3 >= 0.0 && d4 <= d3)
    return b;

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double parameter = d1 / (d1 - d3);
    return add_scaled(a, ab, parameter);
  }

  const double3 cp = subtract(point, c);
  const double d5 = dot_product(ab, cp);
  const double d6 = dot_product(ac, cp);
  if (d6 >= 0.0 && d5 <= d6)
    return c;

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double parameter = d2 / (d2 - d6);
    return add_scaled(a, ac, parameter);
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double parameter =
        (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return add_scaled(b, subtract(c, b), parameter);
  }

  const double inverse = 1.0 / (va + vb + vc);
  const double b_weight = vb * inverse;
  const double c_weight = vc * inverse;
  const double3 ab_term = make_double3(ab.x * b_weight,
                                       ab.y * b_weight,
                                       ab.z * b_weight);
  return add_scaled(make_double3(a.x + ab_term.x, a.y + ab_term.y,
                                 a.z + ab_term.z),
                    ac, c_weight);
}

__device__ int3 axis_neighbour(int index) {
  switch (index) {
  case 0:
    return make_int3(-1, 0, 0);
  case 1:
    return make_int3(1, 0, 0);
  case 2:
    return make_int3(0, -1, 0);
  case 3:
    return make_int3(0, 1, 0);
  case 4:
    return make_int3(0, 0, -1);
  default:
    return make_int3(0, 0, 1);
  }
}

__device__ bool dense_cell_offset(const PackedDenseGrid &grid,
                                  int3 coordinate, size_t &offset) {
  const int x = coordinate.x - grid.minimum.x;
  const int y = coordinate.y - grid.minimum.y;
  const int z = coordinate.z - grid.minimum.z;
  if (x < 0 || y < 0 || z < 0 || x >= grid.dimensions.x ||
      y >= grid.dimensions.y || z >= grid.dimensions.z) {
    return false;
  }
  offset = grid.cell_offset +
           (static_cast<size_t>(x) * grid.dimensions.y + y) *
               grid.dimensions.z +
           z;
  return true;
}

__device__ int3 dense_coordinate(const PackedDenseGrid &grid,
                                 size_t packed_offset) {
  size_t local = packed_offset - grid.cell_offset;
  const size_t yz = static_cast<size_t>(grid.dimensions.y) *
                    grid.dimensions.z;
  const int x = static_cast<int>(local / yz);
  local %= yz;
  const int y = static_cast<int>(local / grid.dimensions.z);
  const int z = static_cast<int>(local % grid.dimensions.z);
  return make_int3(grid.minimum.x + x, grid.minimum.y + y,
                   grid.minimum.z + z);
}

__device__ size_t dense_leaf_offset(const PackedDenseGrid &grid,
                                    int3 coordinate) {
  const int x = (coordinate.x - grid.minimum.x) >> 3;
  const int y = (coordinate.y - grid.minimum.y) >> 3;
  const int z = (coordinate.z - grid.minimum.z) >> 3;
  return grid.leaf_offset +
         (static_cast<size_t>(x) * grid.leaf_dimensions.y + y) *
             grid.leaf_dimensions.z +
         z;
}

__device__ NarrowbandDistance evaluate_dense_candidate(
    const PackedDenseGrid &grid, int3 coordinate, int manhattan_limit,
    const int3 &fragment_minimum, const int3 &fragment_maximum,
    const unsigned char *source_active, const int *triangle_indices,
    const float3 *vertices, const int3 *triangles) {
  double closest_distance = DBL_MAX;
  int closest_triangle_index = 0;
  const double3 point = make_double3(coordinate.x, coordinate.y,
                                     coordinate.z);
  for (int x = fragment_minimum.x; x <= fragment_maximum.x; ++x) {
    for (int y = fragment_minimum.y; y <= fragment_maximum.y; ++y) {
      for (int z = fragment_minimum.z; z <= fragment_maximum.z; ++z) {
        size_t fragment_offset = 0;
        if (!dense_cell_offset(grid, make_int3(x, y, z),
                               fragment_offset) ||
            !source_active[fragment_offset]) {
          continue;
        }
        const int manhattan = abs(x - coordinate.x) +
                              abs(y - coordinate.y) +
                              abs(z - coordinate.z);
        if (manhattan > manhattan_limit)
          continue;
        const int local_triangle_index =
            triangle_indices[fragment_offset];
        const int3 triangle =
            triangles[grid.triangle_offset + local_triangle_index];
        const float3 af = vertices[triangle.x];
        const float3 bf = vertices[triangle.y];
        const float3 cf = vertices[triangle.z];
        const double3 a = make_double3(af.x, af.y, af.z);
        const double3 b = make_double3(bf.x, bf.y, bf.z);
        const double3 c = make_double3(cf.x, cf.y, cf.z);
        const double3 closest = closest_triangle(a, c, b, point);
        const double3 delta = subtract(point, closest);
        const double distance = dot_product(delta, delta);
        if (distance < closest_distance ||
            (distance == closest_distance &&
             local_triangle_index < closest_triangle_index)) {
          closest_distance = distance;
          closest_triangle_index = local_triangle_index;
        }
      }
    }
  }
  return {sqrt(closest_distance) * grid.voxel_size,
          closest_triangle_index};
}

__global__ void construct_dense_mask_kernel(
    const PackedDenseGrid *grids, const int *cell_mesh_indices,
    size_t cell_count, const unsigned char *active, int *mask) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count || !active[cell])
    return;
  const PackedDenseGrid grid = grids[cell_mesh_indices[cell]];
  if (grid.iterations == 0)
    return;
  const int3 coordinate = dense_coordinate(grid, cell);
  for (int neighbour = 0; neighbour < 6; ++neighbour) {
    const int3 delta = axis_neighbour(neighbour);
    const int3 adjacent = make_int3(coordinate.x + delta.x,
                                    coordinate.y + delta.y,
                                    coordinate.z + delta.z);
    size_t adjacent_offset = 0;
    if (dense_cell_offset(grid, adjacent, adjacent_offset) &&
        !active[adjacent_offset]) {
      atomicExch(mask + adjacent_offset, 1);
    }
  }
}

__global__ void reduce_dense_mask_bounds_kernel(
    const PackedDenseGrid *grids, const int *cell_mesh_indices,
    size_t cell_count, unsigned int iteration,
    const unsigned char *active, const int *mask,
    int3 *leaf_minimum, int3 *leaf_maximum) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count || !mask[cell] || active[cell])
    return;
  const PackedDenseGrid grid = grids[cell_mesh_indices[cell]];
  if (iteration >= grid.iterations)
    return;
  const int3 coordinate = dense_coordinate(grid, cell);
  const size_t leaf = dense_leaf_offset(grid, coordinate);
  atomicMin(&leaf_minimum[leaf].x, coordinate.x);
  atomicMin(&leaf_minimum[leaf].y, coordinate.y);
  atomicMin(&leaf_minimum[leaf].z, coordinate.z);
  atomicMax(&leaf_maximum[leaf].x, coordinate.x);
  atomicMax(&leaf_maximum[leaf].y, coordinate.y);
  atomicMax(&leaf_maximum[leaf].z, coordinate.z);
}

__device__ void propagate_first_dense_layer(
    const PackedDenseGrid &grid, int3 coordinate, int *second_mask,
    int *next_mask) {
  const int3 leaf_origin = make_int3(coordinate.x & ~7,
                                     coordinate.y & ~7,
                                     coordinate.z & ~7);
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        const int3 adjacent = make_int3(coordinate.x + dx,
                                        coordinate.y + dy,
                                        coordinate.z + dz);
        size_t adjacent_offset = 0;
        if (!dense_cell_offset(grid, adjacent, adjacent_offset))
          continue;
        const bool same_leaf = (adjacent.x & ~7) == leaf_origin.x &&
                               (adjacent.y & ~7) == leaf_origin.y &&
                               (adjacent.z & ~7) == leaf_origin.z;
        if (same_leaf) {
          atomicExch(second_mask + adjacent_offset, 1);
        } else if (abs(dx) + abs(dy) + abs(dz) == 1) {
          atomicExch(next_mask + adjacent_offset, 1);
        }
      }
    }
  }
}

__global__ void expand_dense_first_layer_kernel(
    const PackedDenseGrid *grids, const int *cell_mesh_indices,
    size_t cell_count, unsigned int iteration, const float3 *vertices,
    const int3 *triangles, const unsigned char *source_active,
    unsigned char *active, const unsigned char *inside,
    double *distances, int *triangle_indices,
    const int *mask, const int3 *leaf_minimum,
    const int3 *leaf_maximum, int *second_mask, int *next_mask) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count || !mask[cell] || active[cell])
    return;
  const PackedDenseGrid grid = grids[cell_mesh_indices[cell]];
  if (iteration >= grid.iterations)
    return;
  const int3 coordinate = dense_coordinate(grid, cell);
  const size_t leaf = dense_leaf_offset(grid, coordinate);
  int3 fragment_minimum = leaf_minimum[leaf];
  int3 fragment_maximum = leaf_maximum[leaf];
  fragment_minimum.x -= 1;
  fragment_minimum.y -= 1;
  fragment_minimum.z -= 1;
  fragment_maximum.x += 1;
  fragment_maximum.y += 1;
  fragment_maximum.z += 1;
  const NarrowbandDistance result = evaluate_dense_candidate(
      grid, coordinate, 5, fragment_minimum, fragment_maximum,
      source_active, triangle_indices, vertices, triangles);
  const bool is_inside = inside[cell] != 0;
  const double width =
      is_inside ? grid.interior_width : grid.exterior_width;
  if (!(result.distance < width))
    return;
  active[cell] = 1;
  distances[cell] = is_inside ? -result.distance : result.distance;
  triangle_indices[cell] = result.triangle_index;
  if (result.distance + grid.voxel_size < width)
    propagate_first_dense_layer(grid, coordinate, second_mask, next_mask);
}

__global__ void expand_dense_second_layer_kernel(
    const PackedDenseGrid *grids, const int *cell_mesh_indices,
    size_t cell_count, unsigned int iteration, const float3 *vertices,
    const int3 *triangles, const unsigned char *source_active,
    unsigned char *active, const unsigned char *inside,
    double *distances, int *triangle_indices,
    const int *second_mask, const int3 *leaf_minimum,
    const int3 *leaf_maximum, int *next_mask) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count || !second_mask[cell] || active[cell])
    return;
  const PackedDenseGrid grid = grids[cell_mesh_indices[cell]];
  if (iteration >= grid.iterations)
    return;
  const int3 coordinate = dense_coordinate(grid, cell);
  const size_t leaf = dense_leaf_offset(grid, coordinate);
  int3 fragment_minimum = leaf_minimum[leaf];
  int3 fragment_maximum = leaf_maximum[leaf];
  fragment_minimum.x -= 1;
  fragment_minimum.y -= 1;
  fragment_minimum.z -= 1;
  fragment_maximum.x += 1;
  fragment_maximum.y += 1;
  fragment_maximum.z += 1;
  const NarrowbandDistance result = evaluate_dense_candidate(
      grid, coordinate, 6, fragment_minimum, fragment_maximum,
      source_active, triangle_indices, vertices, triangles);
  const bool is_inside = inside[cell] != 0;
  const double width =
      is_inside ? grid.interior_width : grid.exterior_width;
  if (!(result.distance < width))
    return;
  active[cell] = 1;
  distances[cell] = is_inside ? -result.distance : result.distance;
  triangle_indices[cell] = result.triangle_index;
  if (result.distance + grid.voxel_size >= width)
    return;
  for (int neighbour = 0; neighbour < 6; ++neighbour) {
    const int3 delta = axis_neighbour(neighbour);
    const int3 adjacent = make_int3(coordinate.x + delta.x,
                                    coordinate.y + delta.y,
                                    coordinate.z + delta.z);
    size_t adjacent_offset = 0;
    if (dense_cell_offset(grid, adjacent, adjacent_offset))
      atomicExch(next_mask + adjacent_offset, 1);
  }
}

__device__ double dense_maximum(double first, double second) {
  return first < second ? second : first;
}

__device__ double dense_minimum(double first, double second) {
  return second < first ? second : first;
}

__device__ double dense_square(double value) { return value * value; }

__device__ double dense_offset_value(
    size_t cell, const unsigned char *active, const double *values,
    double offset) {
  return active[cell] ? values[cell] - offset : values[cell];
}

__device__ double dense_renormalize_neighbour(
    const PackedDenseGrid &grid, int3 coordinate,
    const unsigned char *active, const double *values,
    double offset) {
  size_t neighbour = 0;
  if (!dense_cell_offset(grid, coordinate, neighbour))
    return grid.exterior_width;
  return dense_offset_value(neighbour, active, values, offset);
}

__global__ void renormalize_dense_kernel(
    const PackedDenseGrid *grids, const int *cell_mesh_indices,
    size_t cell_count, const unsigned char *active,
    const double *values, unsigned char *output_active,
    double *output_values) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count)
    return;
  const PackedDenseGrid grid = grids[cell_mesh_indices[cell]];
  if (!grid.renormalize || !active[cell]) {
    output_active[cell] = active[cell];
    output_values[cell] = values[cell];
    return;
  }
  const int3 coordinate = dense_coordinate(grid, cell);
  const double dx = grid.voxel_size;
  const double offset = 0.8 * dx;
  const double phi0 = values[cell] - offset;
  const double down_x =
      phi0 - dense_renormalize_neighbour(
                 grid, make_int3(coordinate.x - 1, coordinate.y,
                                 coordinate.z),
                 active, values, offset);
  const double up_x =
      dense_renormalize_neighbour(
          grid, make_int3(coordinate.x + 1, coordinate.y,
                          coordinate.z),
          active, values, offset) -
      phi0;
  const double down_y =
      phi0 - dense_renormalize_neighbour(
                 grid, make_int3(coordinate.x, coordinate.y - 1,
                                 coordinate.z),
                 active, values, offset);
  const double up_y =
      dense_renormalize_neighbour(
          grid, make_int3(coordinate.x, coordinate.y + 1,
                          coordinate.z),
          active, values, offset) -
      phi0;
  const double down_z =
      phi0 - dense_renormalize_neighbour(
                 grid, make_int3(coordinate.x, coordinate.y,
                                 coordinate.z - 1),
                 active, values, offset);
  const double up_z =
      dense_renormalize_neighbour(
          grid, make_int3(coordinate.x, coordinate.y,
                          coordinate.z + 1),
          active, values, offset) -
      phi0;
  const double zero = 0.0;
  double norm_squared;
  if (phi0 > 0.0) {
    norm_squared = dense_maximum(
        dense_square(dense_maximum(down_x, zero)),
        dense_square(dense_minimum(up_x, zero)));
    norm_squared += dense_maximum(
        dense_square(dense_maximum(down_y, zero)),
        dense_square(dense_minimum(up_y, zero)));
    norm_squared += dense_maximum(
        dense_square(dense_maximum(down_z, zero)),
        dense_square(dense_minimum(up_z, zero)));
  } else {
    norm_squared = dense_maximum(
        dense_square(dense_minimum(down_x, zero)),
        dense_square(dense_maximum(up_x, zero)));
    norm_squared += dense_maximum(
        dense_square(dense_minimum(down_y, zero)),
        dense_square(dense_maximum(up_y, zero)));
    norm_squared += dense_maximum(
        dense_square(dense_minimum(down_z, zero)),
        dense_square(dense_maximum(up_z, zero)));
  }
  const double difference = sqrt(norm_squared) / dx - 1.0;
  const double sign = phi0 / sqrt(phi0 * phi0 + norm_squared);
  const double updated = phi0 - dx * sign * difference;
  double result = dense_minimum(phi0, updated) + offset - 1.0e-7;
  bool remains_active = true;
  if (dense_minimum(grid.interior_width, grid.exterior_width) <
      dx * 4.0) {
    const bool inside = result < 0.0;
    if (inside && !(result > -grid.interior_width)) {
      result = -grid.interior_width;
      remains_active = false;
    } else if (!inside && !(result < grid.exterior_width)) {
      result = grid.exterior_width;
      remains_active = false;
    }
  }
  output_active[cell] = remains_active ? 1 : 0;
  output_values[cell] = result;
}

__global__ void evaluate_narrowband_kernel(
    const float3 *vertices, const int3 *triangles,
    const NarrowbandFragment *fragments,
    const PackedNarrowbandCandidate *candidates, size_t candidate_count,
    NarrowbandDistance *distances) {
  const size_t candidate_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate_index >= candidate_count)
    return;

  const PackedNarrowbandCandidate candidate = candidates[candidate_index];
  const double3 point = make_double3(candidate.x, candidate.y,
                                     candidate.z);
  double closest_distance = DBL_MAX;
  int closest_triangle_index = 0;
  int last_triangle_index = -1;
  const size_t end = candidate.fragment_offset + candidate.fragment_count;
  for (size_t fragment_index = candidate.fragment_offset;
       fragment_index < end; ++fragment_index) {
    const NarrowbandFragment fragment = fragments[fragment_index];
    if (last_triangle_index == fragment.triangle_index)
      continue;
    const int manhattan = abs(fragment.x - candidate.x) +
                          abs(fragment.y - candidate.y) +
                          abs(fragment.z - candidate.z);
    if (manhattan > candidate.manhattan_limit)
      continue;
    last_triangle_index = fragment.triangle_index;

    const int3 triangle = triangles[fragment.triangle_index];
    const float3 af = vertices[triangle.x];
    const float3 bf = vertices[triangle.y];
    const float3 cf = vertices[triangle.z];
    const double3 a = make_double3(af.x, af.y, af.z);
    const double3 b = make_double3(bf.x, bf.y, bf.z);
    const double3 c = make_double3(cf.x, cf.y, cf.z);
    const double3 closest = closest_triangle(a, c, b, point);
    const double3 delta = subtract(point, closest);
    const double distance = dot_product(delta, delta);
    if (distance < closest_distance) {
      closest_distance = distance;
      closest_triangle_index = fragment.triangle_index;
    }
  }
  distances[candidate_index] =
      {sqrt(closest_distance) * candidate.voxel_size,
       closest_triangle_index - candidate.triangle_offset};
}

struct Runtime {
  std::mutex mutex;
  cudaStream_t stream = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer triangles;
  DeviceBuffer fragments;
  DeviceBuffer candidates;
  DeviceBuffer distances;
  PinnedBuffer host_vertices;
  PinnedBuffer host_triangles;
  PinnedBuffer host_fragments;
  PinnedBuffer host_candidates;
  PinnedBuffer host_distances;
  DeviceBuffer dense_grids;
  DeviceBuffer dense_cell_mesh_indices;
  DeviceBuffer dense_active;
  DeviceBuffer dense_source_active;
  DeviceBuffer dense_inside;
  DeviceBuffer dense_distances;
  DeviceBuffer dense_output_distances;
  DeviceBuffer dense_triangle_indices;
  DeviceBuffer dense_mask;
  DeviceBuffer dense_second_mask;
  DeviceBuffer dense_next_mask;
  DeviceBuffer dense_leaf_minimum;
  DeviceBuffer dense_leaf_maximum;
  PinnedBuffer host_dense_grids;
  PinnedBuffer host_dense_cell_mesh_indices;
  PinnedBuffer host_dense_active;
  PinnedBuffer host_dense_inside;
  PinnedBuffer host_dense_distances;
  PinnedBuffer host_dense_triangle_indices;

  ~Runtime() {
    if (stream)
      cudaStreamDestroy(stream);
  }

  void ensure_stream() {
    if (!stream) {
      cuda_memory::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create narrowband CUDA stream");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }
};

Runtime &runtime() {
  static Runtime instance;
  return instance;
}

} // namespace

void evaluate_narrowband_distances_cuda(
    const Mesh &mesh, double scale, double voxel_size,
    const std::vector<NarrowbandFragment> &fragments,
    const std::vector<NarrowbandCandidate> &candidates,
    std::vector<NarrowbandDistance> &distances) {
  std::vector<NarrowbandEvaluationInput> inputs{
      {&mesh, scale, voxel_size, &fragments, &candidates, &distances}};
  evaluate_narrowband_distances_cuda_batch(inputs);
}

void evaluate_narrowband_distances_cuda_batch(
    const std::vector<NarrowbandEvaluationInput> &inputs) {
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  size_t fragment_count = 0;
  size_t candidate_count = 0;
  for (const NarrowbandEvaluationInput &input : inputs) {
    if (!input.mesh || !input.fragments || !input.candidates ||
        !input.distances) {
      throw std::invalid_argument(
          "narrowband CUDA batch contains a null input");
    }
    if (!std::isfinite(input.scale) || input.scale <= 0.0 ||
        !std::isfinite(input.voxel_size) || input.voxel_size <= 0.0) {
      throw std::invalid_argument(
          "narrowband CUDA scale and voxel size must be positive");
    }
    if (input.mesh->vertices.empty() || input.mesh->triangles.empty())
      throw std::invalid_argument("narrowband CUDA mesh is empty");
    if (input.mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                vertex_count ||
        input.mesh->triangles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                triangle_count) {
      throw std::overflow_error("narrowband CUDA packed index overflow");
    }
    vertex_count += input.mesh->vertices.size();
    triangle_count += input.mesh->triangles.size();
    if (input.fragments->size() >
            std::numeric_limits<size_t>::max() - fragment_count ||
        input.candidates->size() >
            std::numeric_limits<size_t>::max() - candidate_count) {
      throw std::overflow_error("narrowband CUDA packed size overflow");
    }
    fragment_count += input.fragments->size();
    candidate_count += input.candidates->size();
    input.distances->resize(input.candidates->size());
  }
  if (candidate_count == 0)
    return;

  Runtime &state = runtime();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_stream();
  state.vertices.ensure(vertex_count * sizeof(float3),
                        "allocate narrowband vertices");
  state.triangles.ensure(triangle_count * sizeof(int3),
                         "allocate narrowband triangles");
  state.fragments.ensure(fragment_count * sizeof(NarrowbandFragment),
                         "allocate narrowband fragments");
  state.candidates.ensure(candidate_count * sizeof(PackedNarrowbandCandidate),
                          "allocate narrowband candidates");
  state.distances.ensure(candidate_count * sizeof(NarrowbandDistance),
                         "allocate narrowband distances");
  state.host_vertices.ensure(vertex_count * sizeof(float3),
                             "allocate host narrowband vertices");
  state.host_triangles.ensure(triangle_count * sizeof(int3),
                              "allocate host narrowband triangles");
  state.host_fragments.ensure(
      fragment_count * sizeof(NarrowbandFragment),
      "allocate host narrowband fragments");
  state.host_candidates.ensure(
      candidate_count * sizeof(PackedNarrowbandCandidate),
      "allocate host narrowband candidates");
  state.host_distances.ensure(
      candidate_count * sizeof(NarrowbandDistance),
      "allocate host narrowband distances");

  float3 *host_vertices = state.host_vertices.as<float3>();
  int3 *host_triangles = state.host_triangles.as<int3>();
  NarrowbandFragment *host_fragments =
      state.host_fragments.as<NarrowbandFragment>();
  PackedNarrowbandCandidate *host_candidates =
      state.host_candidates.as<PackedNarrowbandCandidate>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  size_t fragment_offset = 0;
  size_t candidate_offset = 0;
  for (const NarrowbandEvaluationInput &input : inputs) {
    const Mesh &mesh = *input.mesh;
    for (size_t index = 0; index < mesh.vertices.size(); ++index) {
      const Vec3D &vertex = mesh.vertices[index];
      host_vertices[vertex_offset + index] = make_float3(
          static_cast<float>(vertex[0] * input.scale),
          static_cast<float>(vertex[1] * input.scale),
          static_cast<float>(vertex[2] * input.scale));
    }
    for (size_t index = 0; index < mesh.triangles.size(); ++index) {
      const auto &triangle = mesh.triangles[index];
      host_triangles[triangle_offset + index] = make_int3(
          static_cast<int>(vertex_offset) + triangle[0],
          static_cast<int>(vertex_offset) + triangle[1],
          static_cast<int>(vertex_offset) + triangle[2]);
    }
    for (size_t index = 0; index < input.fragments->size(); ++index) {
      NarrowbandFragment fragment = (*input.fragments)[index];
      fragment.triangle_index += static_cast<int>(triangle_offset);
      host_fragments[fragment_offset + index] = fragment;
    }
    for (size_t index = 0; index < input.candidates->size(); ++index) {
      const NarrowbandCandidate &candidate = (*input.candidates)[index];
      host_candidates[candidate_offset + index] = {
          candidate.x, candidate.y, candidate.z,
          candidate.manhattan_limit,
          fragment_offset + candidate.fragment_offset,
          candidate.fragment_count, static_cast<int>(triangle_offset),
          input.voxel_size};
    }
    vertex_offset += mesh.vertices.size();
    triangle_offset += mesh.triangles.size();
    fragment_offset += input.fragments->size();
    candidate_offset += input.candidates->size();
  }

  cuda_memory::check(
      cudaMemcpyAsync(state.vertices.as<float3>(), host_vertices,
                      vertex_count * sizeof(float3),
                      cudaMemcpyHostToDevice, state.stream),
      "copy narrowband vertices");
  cuda_memory::check(
      cudaMemcpyAsync(state.triangles.as<int3>(), host_triangles,
                      triangle_count * sizeof(int3),
                      cudaMemcpyHostToDevice, state.stream),
      "copy narrowband triangles");
  cuda_memory::check(
      cudaMemcpyAsync(state.fragments.as<NarrowbandFragment>(),
                      state.host_fragments.as<NarrowbandFragment>(),
                      fragment_count * sizeof(NarrowbandFragment),
                      cudaMemcpyHostToDevice, state.stream),
      "copy narrowband fragments");
  cuda_memory::check(
      cudaMemcpyAsync(state.candidates.as<PackedNarrowbandCandidate>(),
                      state.host_candidates.as<PackedNarrowbandCandidate>(),
                      candidate_count * sizeof(PackedNarrowbandCandidate),
                      cudaMemcpyHostToDevice, state.stream),
      "copy narrowband candidates");

  constexpr int threads = 128;
  const int blocks = static_cast<int>(
      (candidate_count + threads - 1) / threads);
  evaluate_narrowband_kernel<<<blocks, threads, 0, state.stream>>>(
      state.vertices.as<float3>(), state.triangles.as<int3>(),
      state.fragments.as<NarrowbandFragment>(),
      state.candidates.as<PackedNarrowbandCandidate>(), candidate_count,
      state.distances.as<NarrowbandDistance>());
  cuda_memory::check(cudaGetLastError(),
                     "launch narrowband distance evaluation");
  cuda_memory::check(
      cudaMemcpyAsync(state.host_distances.as<NarrowbandDistance>(),
                      state.distances.as<NarrowbandDistance>(),
                      candidate_count * sizeof(NarrowbandDistance),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy narrowband distances");
  cuda_memory::check(cudaStreamSynchronize(state.stream),
                     "wait for narrowband distance evaluation");
  candidate_offset = 0;
  for (const NarrowbandEvaluationInput &input : inputs) {
    std::copy_n(state.host_distances.as<NarrowbandDistance>() +
                    candidate_offset,
                input.candidates->size(), input.distances->begin());
    candidate_offset += input.candidates->size();
  }
}

void expand_narrowband_dense_cuda(
    const Mesh &mesh, double scale, DenseNarrowbandGrid &grid) {
  std::vector<DenseNarrowbandInput> inputs{{&mesh, scale, &grid}};
  expand_narrowband_dense_cuda_batch(inputs);
}

void expand_narrowband_dense_cuda_batch(
    const std::vector<DenseNarrowbandInput> &inputs) {
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  size_t cell_count = 0;
  size_t leaf_count = 0;
  unsigned int maximum_iterations = 0;
  for (const DenseNarrowbandInput &input : inputs) {
    if (!input.mesh || !input.grid)
      throw std::invalid_argument(
          "dense narrowband CUDA batch contains a null input");
    DenseNarrowbandGrid &grid = *input.grid;
    if (!std::isfinite(input.scale) || input.scale <= 0.0 ||
        !std::isfinite(grid.exterior_width) ||
        !std::isfinite(grid.interior_width) ||
        !std::isfinite(grid.voxel_size) || grid.exterior_width <= 0.0 ||
        grid.interior_width <= 0.0 || grid.voxel_size <= 0.0) {
      throw std::invalid_argument(
          "dense narrowband CUDA parameters must be finite and positive");
    }
    if (input.mesh->vertices.empty() || input.mesh->triangles.empty())
      throw std::invalid_argument("dense narrowband CUDA mesh is empty");
    if ((grid.minimum[0] & 7) != 0 || (grid.minimum[1] & 7) != 0 ||
        (grid.minimum[2] & 7) != 0 || grid.dimensions[0] <= 0 ||
        grid.dimensions[1] <= 0 || grid.dimensions[2] <= 0 ||
        (grid.dimensions[0] & 7) != 0 ||
        (grid.dimensions[1] & 7) != 0 ||
        (grid.dimensions[2] & 7) != 0) {
      throw std::invalid_argument(
          "dense narrowband CUDA bounds must contain complete leaves");
    }
    size_t grid_cells = static_cast<size_t>(grid.dimensions[0]);
    if (grid_cells > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(grid.dimensions[1]))
      throw std::overflow_error("dense narrowband CUDA cell overflow");
    grid_cells *= static_cast<size_t>(grid.dimensions[1]);
    if (grid_cells > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(grid.dimensions[2]))
      throw std::overflow_error("dense narrowband CUDA cell overflow");
    grid_cells *= static_cast<size_t>(grid.dimensions[2]);
    if (grid.active.size() != grid_cells ||
        grid.inside.size() != grid_cells ||
        grid.distances.size() != grid_cells ||
        grid.triangle_indices.size() != grid_cells) {
      throw std::invalid_argument(
          "dense narrowband CUDA arrays do not match their bounds");
    }
    const size_t grid_leaves = grid_cells / (8 * 8 * 8);
    if (input.mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.mesh->triangles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("dense narrowband CUDA mesh overflow");
    }
    if (cell_count > std::numeric_limits<size_t>::max() - grid_cells ||
        leaf_count > std::numeric_limits<size_t>::max() - grid_leaves ||
        vertex_count > static_cast<size_t>(
                           std::numeric_limits<int>::max()) -
                           input.mesh->vertices.size() ||
        triangle_count > static_cast<size_t>(
                             std::numeric_limits<int>::max()) -
                             input.mesh->triangles.size()) {
      throw std::overflow_error("dense narrowband CUDA packed overflow");
    }
    cell_count += grid_cells;
    leaf_count += grid_leaves;
    vertex_count += input.mesh->vertices.size();
    triangle_count += input.mesh->triangles.size();
    maximum_iterations = std::max(maximum_iterations, grid.iterations);
  }
  if (inputs.empty() || cell_count == 0 || maximum_iterations == 0)
    return;
  if (inputs.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      cell_count > std::numeric_limits<size_t>::max() / sizeof(double)) {
    throw std::overflow_error("dense narrowband CUDA allocation overflow");
  }
  constexpr size_t threads = 128;
  if ((cell_count + threads - 1) / threads >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("dense narrowband CUDA launch overflow");
  }

  Runtime &state = runtime();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_stream();
  state.vertices.ensure(vertex_count * sizeof(float3),
                        "allocate dense narrowband vertices");
  state.triangles.ensure(triangle_count * sizeof(int3),
                         "allocate dense narrowband triangles");
  state.dense_grids.ensure(inputs.size() * sizeof(PackedDenseGrid),
                           "allocate dense narrowband grids");
  state.dense_cell_mesh_indices.ensure(
      cell_count * sizeof(int), "allocate dense narrowband owners");
  state.dense_active.ensure(cell_count * sizeof(unsigned char),
                            "allocate dense narrowband activity");
  state.dense_source_active.ensure(
      cell_count * sizeof(unsigned char),
      "allocate dense narrowband source activity");
  state.dense_inside.ensure(cell_count * sizeof(unsigned char),
                            "allocate dense narrowband signs");
  state.dense_distances.ensure(cell_count * sizeof(double),
                               "allocate dense narrowband distances");
  state.dense_output_distances.ensure(
      cell_count * sizeof(double),
      "allocate dense narrowband output distances");
  state.dense_triangle_indices.ensure(
      cell_count * sizeof(int), "allocate dense narrowband indices");
  state.dense_mask.ensure(cell_count * sizeof(int),
                          "allocate dense narrowband mask");
  state.dense_second_mask.ensure(
      cell_count * sizeof(int), "allocate dense narrowband second mask");
  state.dense_next_mask.ensure(
      cell_count * sizeof(int), "allocate dense narrowband next mask");
  state.dense_leaf_minimum.ensure(
      leaf_count * sizeof(int3), "allocate dense narrowband leaf minima");
  state.dense_leaf_maximum.ensure(
      leaf_count * sizeof(int3), "allocate dense narrowband leaf maxima");
  state.host_vertices.ensure(vertex_count * sizeof(float3),
                             "allocate host dense narrowband vertices");
  state.host_triangles.ensure(triangle_count * sizeof(int3),
                              "allocate host dense narrowband triangles");
  state.host_dense_grids.ensure(
      inputs.size() * sizeof(PackedDenseGrid),
      "allocate host dense narrowband grids");
  state.host_dense_cell_mesh_indices.ensure(
      cell_count * sizeof(int), "allocate host dense narrowband owners");
  state.host_dense_active.ensure(
      cell_count * sizeof(unsigned char),
      "allocate host dense narrowband activity");
  state.host_dense_inside.ensure(
      cell_count * sizeof(unsigned char),
      "allocate host dense narrowband signs");
  state.host_dense_distances.ensure(
      cell_count * sizeof(double), "allocate host dense narrowband distances");
  state.host_dense_triangle_indices.ensure(
      cell_count * sizeof(int), "allocate host dense narrowband indices");

  float3 *host_vertices = state.host_vertices.as<float3>();
  int3 *host_triangles = state.host_triangles.as<int3>();
  PackedDenseGrid *host_grids =
      state.host_dense_grids.as<PackedDenseGrid>();
  int *host_owners = state.host_dense_cell_mesh_indices.as<int>();
  unsigned char *host_active =
      state.host_dense_active.as<unsigned char>();
  unsigned char *host_inside =
      state.host_dense_inside.as<unsigned char>();
  double *host_grid_distances =
      state.host_dense_distances.as<double>();
  int *host_grid_indices =
      state.host_dense_triangle_indices.as<int>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  size_t cell_offset = 0;
  size_t leaf_offset = 0;
  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    const DenseNarrowbandInput &input = inputs[input_index];
    const Mesh &mesh = *input.mesh;
    DenseNarrowbandGrid &grid = *input.grid;
    const size_t grid_cells = grid.active.size();
    const int3 leaf_dimensions = make_int3(
        grid.dimensions[0] / 8, grid.dimensions[1] / 8,
        grid.dimensions[2] / 8);
    host_grids[input_index] = {
        make_int3(grid.minimum[0], grid.minimum[1], grid.minimum[2]),
        make_int3(grid.dimensions[0], grid.dimensions[1],
                  grid.dimensions[2]),
        leaf_dimensions, cell_offset, grid_cells, leaf_offset,
        vertex_offset, triangle_offset, grid.exterior_width,
        grid.interior_width, grid.voxel_size, grid.iterations,
        grid.renormalize ? 1 : 0};
    for (size_t index = 0; index < mesh.vertices.size(); ++index) {
      const Vec3D &vertex = mesh.vertices[index];
      host_vertices[vertex_offset + index] = make_float3(
          static_cast<float>(vertex[0] * input.scale),
          static_cast<float>(vertex[1] * input.scale),
          static_cast<float>(vertex[2] * input.scale));
    }
    for (size_t index = 0; index < mesh.triangles.size(); ++index) {
      const auto &triangle = mesh.triangles[index];
      host_triangles[triangle_offset + index] = make_int3(
          static_cast<int>(vertex_offset) + triangle[0],
          static_cast<int>(vertex_offset) + triangle[1],
          static_cast<int>(vertex_offset) + triangle[2]);
    }
    for (size_t index = 0; index < grid_cells; ++index) {
      const size_t packed = cell_offset + index;
      host_owners[packed] = static_cast<int>(input_index);
      host_active[packed] = grid.active[index] ? 1 : 0;
      host_inside[packed] = grid.inside[index] ? 1 : 0;
      host_grid_distances[packed] = grid.distances[index];
      host_grid_indices[packed] = grid.triangle_indices[index];
    }
    vertex_offset += mesh.vertices.size();
    triangle_offset += mesh.triangles.size();
    cell_offset += grid_cells;
    leaf_offset += static_cast<size_t>(leaf_dimensions.x) *
                   leaf_dimensions.y * leaf_dimensions.z;
  }

  const auto copy_to_device = [&](void *destination, const void *source,
                                  size_t bytes, const char *message) {
    cuda_memory::check(cudaMemcpyAsync(destination, source, bytes,
                                       cudaMemcpyHostToDevice,
                                       state.stream),
                       message);
  };
  copy_to_device(state.vertices.as<float3>(), host_vertices,
                 vertex_count * sizeof(float3),
                 "copy dense narrowband vertices");
  copy_to_device(state.triangles.as<int3>(), host_triangles,
                 triangle_count * sizeof(int3),
                 "copy dense narrowband triangles");
  copy_to_device(state.dense_grids.as<PackedDenseGrid>(), host_grids,
                 inputs.size() * sizeof(PackedDenseGrid),
                 "copy dense narrowband grids");
  copy_to_device(state.dense_cell_mesh_indices.as<int>(), host_owners,
                 cell_count * sizeof(int),
                 "copy dense narrowband owners");
  copy_to_device(state.dense_active.as<unsigned char>(), host_active,
                 cell_count * sizeof(unsigned char),
                 "copy dense narrowband activity");
  copy_to_device(state.dense_inside.as<unsigned char>(), host_inside,
                 cell_count * sizeof(unsigned char),
                 "copy dense narrowband signs");
  copy_to_device(state.dense_distances.as<double>(), host_grid_distances,
                 cell_count * sizeof(double),
                 "copy dense narrowband distances");
  copy_to_device(state.dense_triangle_indices.as<int>(), host_grid_indices,
                 cell_count * sizeof(int),
                 "copy dense narrowband indices");
  cuda_memory::check(
      cudaMemsetAsync(state.dense_mask.as<int>(), 0,
                      cell_count * sizeof(int), state.stream),
      "clear dense narrowband mask");
  const int blocks =
      static_cast<int>((cell_count + threads - 1) / threads);
  construct_dense_mask_kernel<<<blocks, threads, 0, state.stream>>>(
      state.dense_grids.as<PackedDenseGrid>(),
      state.dense_cell_mesh_indices.as<int>(), cell_count,
      state.dense_active.as<unsigned char>(),
      state.dense_mask.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "construct dense narrowband mask");

  for (unsigned int iteration = 0; iteration < maximum_iterations;
       ++iteration) {
    cuda_memory::check(
        cudaMemsetAsync(state.dense_leaf_minimum.as<int3>(), 0x7f,
                        leaf_count * sizeof(int3), state.stream),
        "clear dense narrowband leaf minima");
    cuda_memory::check(
        cudaMemsetAsync(state.dense_leaf_maximum.as<int3>(), 0x80,
                        leaf_count * sizeof(int3), state.stream),
        "clear dense narrowband leaf maxima");
    cuda_memory::check(
        cudaMemsetAsync(state.dense_second_mask.as<int>(), 0,
                        cell_count * sizeof(int), state.stream),
        "clear dense narrowband second mask");
    cuda_memory::check(
        cudaMemsetAsync(state.dense_next_mask.as<int>(), 0,
                        cell_count * sizeof(int), state.stream),
        "clear dense narrowband next mask");
    cuda_memory::check(
        cudaMemcpyAsync(state.dense_source_active.as<unsigned char>(),
                        state.dense_active.as<unsigned char>(),
                        cell_count * sizeof(unsigned char),
                        cudaMemcpyDeviceToDevice, state.stream),
        "snapshot dense narrowband activity");
    reduce_dense_mask_bounds_kernel<<<blocks, threads, 0, state.stream>>>(
        state.dense_grids.as<PackedDenseGrid>(),
        state.dense_cell_mesh_indices.as<int>(), cell_count, iteration,
        state.dense_active.as<unsigned char>(),
        state.dense_mask.as<int>(),
        state.dense_leaf_minimum.as<int3>(),
        state.dense_leaf_maximum.as<int3>());
    cuda_memory::check(cudaGetLastError(),
                       "reduce dense narrowband mask bounds");
    expand_dense_first_layer_kernel<<<blocks, threads, 0, state.stream>>>(
        state.dense_grids.as<PackedDenseGrid>(),
        state.dense_cell_mesh_indices.as<int>(), cell_count, iteration,
        state.vertices.as<float3>(), state.triangles.as<int3>(),
        state.dense_source_active.as<unsigned char>(),
        state.dense_active.as<unsigned char>(),
        state.dense_inside.as<unsigned char>(),
        state.dense_distances.as<double>(),
        state.dense_triangle_indices.as<int>(), state.dense_mask.as<int>(),
        state.dense_leaf_minimum.as<int3>(),
        state.dense_leaf_maximum.as<int3>(),
        state.dense_second_mask.as<int>(),
        state.dense_next_mask.as<int>());
    cuda_memory::check(cudaGetLastError(),
                       "expand dense narrowband first layer");
    expand_dense_second_layer_kernel<<<blocks, threads, 0, state.stream>>>(
        state.dense_grids.as<PackedDenseGrid>(),
        state.dense_cell_mesh_indices.as<int>(), cell_count, iteration,
        state.vertices.as<float3>(), state.triangles.as<int3>(),
        state.dense_source_active.as<unsigned char>(),
        state.dense_active.as<unsigned char>(),
        state.dense_inside.as<unsigned char>(),
        state.dense_distances.as<double>(),
        state.dense_triangle_indices.as<int>(),
        state.dense_second_mask.as<int>(),
        state.dense_leaf_minimum.as<int3>(),
        state.dense_leaf_maximum.as<int3>(),
        state.dense_next_mask.as<int>());
    cuda_memory::check(cudaGetLastError(),
                       "expand dense narrowband second layer");
    cuda_memory::check(
        cudaMemcpyAsync(state.dense_mask.as<int>(),
                        state.dense_next_mask.as<int>(),
                        cell_count * sizeof(int), cudaMemcpyDeviceToDevice,
                        state.stream),
        "advance dense narrowband mask");
  }

  renormalize_dense_kernel<<<blocks, threads, 0, state.stream>>>(
      state.dense_grids.as<PackedDenseGrid>(),
      state.dense_cell_mesh_indices.as<int>(), cell_count,
      state.dense_active.as<unsigned char>(),
      state.dense_distances.as<double>(),
      state.dense_source_active.as<unsigned char>(),
      state.dense_output_distances.as<double>());
  cuda_memory::check(cudaGetLastError(),
                     "renormalize dense narrowband");

  std::vector<DenseVolumeMeshingGrid> meshing_storage;
  std::vector<DenseVolumeMeshingGrid *> meshing_grids;
  std::vector<DenseNarrowbandGrid *> meshing_outputs;
  std::vector<size_t> meshing_cell_offsets;
  meshing_storage.reserve(inputs.size());
  meshing_grids.reserve(inputs.size());
  meshing_outputs.reserve(inputs.size());
  meshing_cell_offsets.reserve(inputs.size());
  bool readback = false;
  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    DenseNarrowbandGrid &grid = *inputs[input_index].grid;
    if (!grid.mesh_output) {
      readback = true;
      continue;
    }
    meshing_storage.emplace_back();
    DenseVolumeMeshingGrid &mesh_grid = meshing_storage.back();
    for (int axis = 0; axis < 3; ++axis) {
      mesh_grid.minimum[axis] = grid.minimum[axis];
      mesh_grid.dimensions[axis] = grid.dimensions[axis];
    }
    mesh_grid.isovalue = grid.isovalue;
    mesh_grid.leaf_order = grid.leaf_order;
    mesh_grid.retain_device_mesh = grid.retain_device_mesh;
    mesh_grid.output_scale = grid.output_scale;
    mesh_grid.device_memory_fraction = grid.device_memory_fraction;
    meshing_grids.push_back(&mesh_grid);
    meshing_outputs.push_back(&grid);
    meshing_cell_offsets.push_back(host_grids[input_index].cell_offset);
  }
  if (!meshing_grids.empty()) {
    mesh_dense_volume_cuda_device_batch(
        meshing_grids,
        state.dense_source_active.as<unsigned char>(),
        state.dense_output_distances.as<double>(),
        meshing_cell_offsets, reinterpret_cast<void *>(state.stream));
    for (size_t index = 0; index < meshing_grids.size(); ++index) {
      meshing_outputs[index]->points =
          std::move(meshing_grids[index]->points);
      meshing_outputs[index]->quads =
          std::move(meshing_grids[index]->quads);
      meshing_outputs[index]->device_mesh =
          std::move(meshing_grids[index]->device_mesh);
    }
  }
  if (!readback)
    return;

  cuda_memory::check(
      cudaMemcpyAsync(host_active,
                      state.dense_source_active.as<unsigned char>(),
                      cell_count * sizeof(unsigned char),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy dense narrowband activity");
  cuda_memory::check(
      cudaMemcpyAsync(host_grid_distances,
                      state.dense_output_distances.as<double>(),
                      cell_count * sizeof(double), cudaMemcpyDeviceToHost,
                      state.stream),
      "copy dense narrowband distances");
  cuda_memory::check(
      cudaMemcpyAsync(host_grid_indices,
                      state.dense_triangle_indices.as<int>(),
                      cell_count * sizeof(int), cudaMemcpyDeviceToHost,
                      state.stream),
      "copy dense narrowband indices");
  cuda_memory::check(cudaStreamSynchronize(state.stream),
                     "wait for dense narrowband expansion");

  cell_offset = 0;
  for (const DenseNarrowbandInput &input : inputs) {
    DenseNarrowbandGrid &grid = *input.grid;
    for (size_t index = 0; index < grid.active.size(); ++index) {
      const size_t packed = cell_offset + index;
      grid.active[index] = host_active[packed] ? 1 : 0;
      grid.distances[index] = host_grid_distances[packed];
      grid.triangle_indices[index] = host_grid_indices[packed];
    }
    cell_offset += grid.active.size();
  }
}

} // namespace neural_acd
