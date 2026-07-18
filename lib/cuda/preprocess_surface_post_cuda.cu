#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <preprocess_surface_post_cuda.hpp>

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
constexpr size_t kLeafNeighbourCount = 3 * 3 * 3;
constexpr size_t kAxisNeighbourCount = 6;
constexpr int kInvalidIndex = -1;

struct PackedSurfacePostGrid {
  size_t triangle_offset;
  double voxel_size;
};

struct CellData {
  double value;
  int triangle;
  bool active;
};

__device__ bool trace_voxel_line(double *values, int leaf, int position,
                                 int step) {
  bool outside = true;
  const size_t base = static_cast<size_t>(leaf) * kLeafSize;
  for (int index = 0; index < 8; ++index) {
    double &distance = values[base + position];
    if (distance < 0.0) {
      outside = true;
    } else {
      if (!(distance > 0.75))
        outside = false;
      if (outside)
        distance = -distance;
    }
    position += step;
  }
  return outside;
}

__global__ void sweep_exterior_kernel(
    size_t leaf_count, int axis, const int *axis_neighbours,
    double *values) {
  const size_t line =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (line >= leaf_count * 64)
    return;
  const int start = static_cast<int>(line / 64);
  const int lane = static_cast<int>(line & 63);
  const int previous_slot = axis * 2 + 1;
  const int next_slot = axis * 2;
  if (axis_neighbours[static_cast<size_t>(start) *
                          kAxisNeighbourCount +
                      previous_slot] != kInvalidIndex)
    return;

  int step;
  int position;
  if (axis == 2) {
    step = 1;
    position = (lane / 8) * 64 + (lane & 7) * 8;
  } else if (axis == 1) {
    step = 8;
    position = (lane / 8) * 64 + (lane & 7);
  } else {
    step = 64;
    position = (lane / 8) * 8 + (lane & 7);
  }

  int offset = start;
  int last = start;
  while (offset != kInvalidIndex &&
         trace_voxel_line(values, offset, position, step)) {
    last = offset;
    offset = axis_neighbours[static_cast<size_t>(offset) *
                                 kAxisNeighbourCount +
                             next_slot];
  }
  offset = last;
  while (offset != kInvalidIndex) {
    last = offset;
    offset = axis_neighbours[static_cast<size_t>(offset) *
                                 kAxisNeighbourCount +
                             next_slot];
  }
  offset = last;
  position += step * 7;
  while (offset != kInvalidIndex &&
         trace_voxel_line(values, offset, position, -step)) {
    offset = axis_neighbours[static_cast<size_t>(offset) *
                                 kAxisNeighbourCount +
                             previous_slot];
  }
}

__global__ void scan_fill_exterior_kernel(
    size_t leaf_count, const unsigned char *changed_nodes,
    double *values) {
  const size_t leaf =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (leaf >= leaf_count || !changed_nodes[leaf])
    return;
  double *data = values + leaf * kLeafSize;
  bool changed;
  do {
    changed = false;
    for (int position = 0; position < 512; ++position) {
      double &distance = data[position];
      if (distance < 0.0 || !(distance > 0.75))
        continue;
      const int x = position / 64;
      const int y = (position / 8) & 7;
      const int z = position & 7;
      if ((z != 0 && data[position - 1] < 0.0) ||
          (z != 7 && data[position + 1] < 0.0) ||
          (y != 0 && data[position - 8] < 0.0) ||
          (y != 7 && data[position + 8] < 0.0) ||
          (x != 0 && data[position - 64] < 0.0) ||
          (x != 7 && data[position + 64] < 0.0)) {
        distance = -distance;
        changed = true;
      }
    }
  } while (changed);
}

__device__ bool seed_exterior_face(
    size_t leaf, int axis, bool first_face,
    const int *axis_neighbours, const unsigned char *changed_nodes,
    const double *values, unsigned char *changed_voxels) {
  const int slot = axis * 2 + (first_face ? 1 : 0);
  const int neighbour =
      axis_neighbours[leaf * kAxisNeighbourCount + slot];
  if (neighbour == kInvalidIndex || !changed_nodes[neighbour])
    return false;
  const double *lhs = values + leaf * kLeafSize;
  const double *rhs =
      values + static_cast<size_t>(neighbour) * kLeafSize;
  unsigned char *mask = changed_voxels + leaf * kLeafSize;
  bool changed = false;
  for (int first = 0; first < 8; ++first) {
    for (int second = 0; second < 8; ++second) {
      int lhs_position;
      int rhs_position;
      if (axis == 2) {
        const int base = first * 64 + second * 8;
        lhs_position = base + (first_face ? 0 : 7);
        rhs_position = base + (first_face ? 7 : 0);
      } else if (axis == 1) {
        const int base = first * 64 + second;
        lhs_position = base + (first_face ? 0 : 56);
        rhs_position = base + (first_face ? 56 : 0);
      } else {
        const int base = first * 8 + second;
        lhs_position = base + (first_face ? 0 : 448);
        rhs_position = base + (first_face ? 448 : 0);
      }
      if (lhs[lhs_position] > 0.75 && rhs[rhs_position] < 0.0) {
        mask[lhs_position] = 1;
        changed = true;
      }
    }
  }
  return changed;
}

__global__ void seed_exterior_points_kernel(
    size_t leaf_count, const int *axis_neighbours,
    const unsigned char *changed_nodes,
    unsigned char *next_changed_nodes,
    const double *values, unsigned char *changed_voxels,
    int *any_changed) {
  const size_t leaf =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (leaf >= leaf_count)
    return;
  bool changed = false;
  changed |= seed_exterior_face(leaf, 2, true, axis_neighbours,
                                changed_nodes, values,
                                changed_voxels);
  changed |= seed_exterior_face(leaf, 2, false, axis_neighbours,
                                changed_nodes, values,
                                changed_voxels);
  changed |= seed_exterior_face(leaf, 1, true, axis_neighbours,
                                changed_nodes, values,
                                changed_voxels);
  changed |= seed_exterior_face(leaf, 1, false, axis_neighbours,
                                changed_nodes, values,
                                changed_voxels);
  changed |= seed_exterior_face(leaf, 0, true, axis_neighbours,
                                changed_nodes, values,
                                changed_voxels);
  changed |= seed_exterior_face(leaf, 0, false, axis_neighbours,
                                changed_nodes, values,
                                changed_voxels);
  next_changed_nodes[leaf] = changed ? 1 : 0;
  if (changed)
    atomicExch(any_changed, 1);
}

__global__ void sync_exterior_voxels_kernel(
    size_t leaf_count, const unsigned char *changed_nodes,
    unsigned char *changed_voxels, double *values) {
  const size_t leaf =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (leaf >= leaf_count || !changed_nodes[leaf])
    return;
  double *data = values + leaf * kLeafSize;
  unsigned char *mask = changed_voxels + leaf * kLeafSize;
  for (int position = 0; position < 512; ++position) {
    if (mask[position]) {
      data[position] = -data[position];
      mask[position] = 0;
    }
  }
}

__device__ double3 subtract(double3 first, double3 second) {
  return make_double3(first.x - second.x, first.y - second.y,
                      first.z - second.z);
}

__device__ double dot_product(double3 first, double3 second) {
  return first.x * second.x + first.y * second.y +
         first.z * second.z;
}

__device__ double3 add_scaled(double3 point, double3 direction,
                              double scale) {
  return make_double3(point.x + direction.x * scale,
                      point.y + direction.y * scale,
                      point.z + direction.z * scale);
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
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    return add_scaled(a, ab, d1 / (d1 - d3));
  const double3 cp = subtract(point, c);
  const double d5 = dot_product(ab, cp);
  const double d6 = dot_product(ac, cp);
  if (d6 >= 0.0 && d5 <= d6)
    return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    return add_scaled(a, ac, d2 / (d2 - d6));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double parameter =
        (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return add_scaled(b, subtract(c, b), parameter);
  }
  const double inverse = 1.0 / (va + vb + vc);
  const double b_weight = vb * inverse;
  const double c_weight = vc * inverse;
  const double3 ab_term = make_double3(
      ab.x * b_weight, ab.y * b_weight, ab.z * b_weight);
  return add_scaled(
      make_double3(a.x + ab_term.x, a.y + ab_term.y,
                   a.z + ab_term.z),
      ac, c_weight);
}

__device__ double3 normalize_like_openvdb(double3 vector) {
  const double length = sqrt(dot_product(vector, vector));
  if (!(fabs(length) > 1.0e-7))
    return vector;
  const double inverse = 1.0 / length;
  return make_double3(vector.x * inverse, vector.y * inverse,
                      vector.z * inverse);
}

__device__ int3 neighbour_offset(int index) {
  const int offsets[26][3] = {
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
      {0, 0, 1}, {0, 0, -1}, {1, 0, -1}, {-1, 0, -1},
      {1, 0, 1}, {-1, 0, 1}, {1, 1, 0}, {-1, 1, 0},
      {1, -1, 0}, {-1, -1, 0}, {0, -1, 1}, {0, -1, -1},
      {0, 1, 1}, {0, 1, -1}, {-1, -1, -1}, {-1, -1, 1},
      {1, -1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, 1, 1},
      {1, 1, 1}, {1, 1, -1}};
  return make_int3(offsets[index][0], offsets[index][1],
                   offsets[index][2]);
}

__device__ CellData read_cell(
    size_t leaf, int x, int y, int z, const int *neighbour_indices,
    const double *neighbour_values, const unsigned char *active,
    const double *values, const int *triangle_indices) {
  int leaf_x = 0, leaf_y = 0, leaf_z = 0;
  if (x < 0) {
    leaf_x = -1;
    x += 8;
  } else if (x >= 8) {
    leaf_x = 1;
    x -= 8;
  }
  if (y < 0) {
    leaf_y = -1;
    y += 8;
  } else if (y >= 8) {
    leaf_y = 1;
    y -= 8;
  }
  if (z < 0) {
    leaf_z = -1;
    z += 8;
  } else if (z >= 8) {
    leaf_z = 1;
    z -= 8;
  }
  const size_t slot =
      static_cast<size_t>(leaf_x + 1) * 9 +
      static_cast<size_t>(leaf_y + 1) * 3 + leaf_z + 1;
  const size_t neighbour = leaf * kLeafNeighbourCount + slot;
  const int packed_leaf = neighbour_indices[neighbour];
  if (packed_leaf < 0)
    return {neighbour_values[neighbour], kInvalidIndex, false};
  const size_t cell =
      static_cast<size_t>(packed_leaf) * kLeafSize +
      (static_cast<size_t>(x) * 8 + y) * 8 + z;
  return {values[cell], triangle_indices[cell], active[cell] != 0};
}

__device__ double3 closest_point(
    const PackedSurfacePostGrid &grid, int triangle_index,
    double3 point, const float3 *vertices, const int3 *triangles) {
  const int3 triangle =
      triangles[grid.triangle_offset + triangle_index];
  const float3 af = vertices[triangle.x];
  const float3 bf = vertices[triangle.y];
  const float3 cf = vertices[triangle.z];
  const double3 a = make_double3(af.x, af.y, af.z);
  const double3 b = make_double3(bf.x, bf.y, bf.z);
  const double3 c = make_double3(cf.x, cf.y, cf.z);
  return closest_triangle(a, c, b, point);
}

__global__ void compute_surface_sign_kernel(
    const PackedSurfacePostGrid *grids, const int *leaf_grid_indices,
    const int3 *leaf_origins, size_t cell_count,
    const int *neighbour_indices, const double *neighbour_values,
    const unsigned char *active, const double *values,
    const int *triangle_indices, const float3 *vertices,
    const int3 *triangles, double *output) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count)
    return;
  const double distance = values[cell];
  if (!active[cell] || distance < 0.0 || distance > 0.75) {
    output[cell] = distance;
    return;
  }
  const size_t leaf = cell / kLeafSize;
  const size_t local = cell - leaf * kLeafSize;
  const int x = static_cast<int>(local / 64);
  const int y = static_cast<int>((local / 8) & 7);
  const int z = static_cast<int>(local & 7);
  const int3 origin = leaf_origins[leaf];
  const double3 point = make_double3(
      origin.x + x, origin.y + y, origin.z + z);
  const PackedSurfacePostGrid grid =
      grids[leaf_grid_indices[leaf]];
  bool flip = false;
  for (int nx = max(0, x - 1); nx <= min(7, x + 1) && !flip; ++nx) {
    for (int ny = max(0, y - 1); ny <= min(7, y + 1) && !flip; ++ny) {
      for (int nz = max(0, z - 1); nz <= min(7, z + 1); ++nz) {
        const size_t neighbour_cell =
            leaf * kLeafSize +
            (static_cast<size_t>(nx) * 8 + ny) * 8 + nz;
        const int triangle_index = triangle_indices[neighbour_cell];
        if (triangle_index == kInvalidIndex ||
            !(values[neighbour_cell] < -0.75))
          continue;
        const double3 neighbour_point = make_double3(
            origin.x + nx, origin.y + ny, origin.z + nz);
        const double3 closest = closest_point(
            grid, triangle_index, neighbour_point, vertices, triangles);
        const double3 outward = normalize_like_openvdb(
            subtract(neighbour_point, closest));
        const double3 candidate = normalize_like_openvdb(
            subtract(point, closest));
        if (dot_product(outward, candidate) > 0.0) {
          flip = true;
          break;
        }
      }
    }
  }
  for (int index = 0; index < 26 && !flip; ++index) {
    const int3 delta = neighbour_offset(index);
    const int nx = x + delta.x;
    const int ny = y + delta.y;
    const int nz = z + delta.z;
    if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8 &&
        nz >= 0 && nz < 8)
      continue;
    const CellData adjacent = read_cell(
        leaf, nx, ny, nz, neighbour_indices, neighbour_values,
        active, values, triangle_indices);
    if (!adjacent.active || !(adjacent.value < -0.75))
      continue;
    const double3 neighbour_point = make_double3(
        point.x + delta.x, point.y + delta.y, point.z + delta.z);
    const double3 closest = closest_point(
        grid, adjacent.triangle, neighbour_point, vertices, triangles);
    const double3 candidate = normalize_like_openvdb(
        subtract(point, closest));
    const double3 outward = normalize_like_openvdb(
        subtract(neighbour_point, closest));
    if (dot_product(outward, candidate) > 0.0)
      flip = true;
  }
  output[cell] = flip ? -distance : distance;
}

__device__ bool matching_neighbour(
    size_t leaf, int x, int y, int z, const int *neighbour_indices,
    const double *neighbour_values, const unsigned char *active,
    const double *values, const int *triangle_indices,
    bool negative) {
  for (int index = 0; index < 26; ++index) {
    const int3 delta = neighbour_offset(index);
    const int nx = x + delta.x;
    const int ny = y + delta.y;
    const int nz = z + delta.z;
    double value;
    if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8 &&
        nz >= 0 && nz < 8) {
      int local_x = nx, local_y = ny, local_z = nz;
      if (index == 6)
        local_z = z;
      const size_t adjacent =
          leaf * kLeafSize +
          (static_cast<size_t>(local_x) * 8 + local_y) * 8 +
          local_z;
      value = values[adjacent];
    } else {
      value = read_cell(
                  leaf, nx, ny, nz, neighbour_indices,
                  neighbour_values, active, values,
                  triangle_indices)
                  .value;
    }
    if (negative ? value < 0.0 : !(value > 0.75))
      return true;
  }
  return false;
}

__global__ void validate_surface_kernel(
    const int *leaf_grid_indices, size_t cell_count,
    const int *neighbour_indices, const double *neighbour_values,
    const unsigned char *active, const double *values,
    const int *triangle_indices, double *output) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count)
    return;
  const double distance = values[cell];
  if (!active[cell] || distance < 0.0 || distance > 0.75) {
    output[cell] = distance;
    return;
  }
  const size_t leaf = cell / kLeafSize;
  const size_t local = cell - leaf * kLeafSize;
  const int x = static_cast<int>(local / 64);
  const int y = static_cast<int>((local / 8) & 7);
  const int z = static_cast<int>(local & 7);
  const bool has_negative = matching_neighbour(
      leaf, x, y, z, neighbour_indices, neighbour_values, active,
      values, triangle_indices, true);
  output[cell] = has_negative ? distance : 0.7500001;
}

__global__ void cleanup_surface_kernel(
    size_t cell_count, const int *neighbour_indices,
    const double *neighbour_values, const unsigned char *active,
    const double *values, const int *triangle_indices,
    unsigned char *output) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count)
    return;
  if (!active[cell] || !(values[cell] > 0.75)) {
    output[cell] = active[cell];
    return;
  }
  const size_t leaf = cell / kLeafSize;
  const size_t local = cell - leaf * kLeafSize;
  const int x = static_cast<int>(local / 64);
  const int y = static_cast<int>((local / 8) & 7);
  const int z = static_cast<int>(local & 7);
  output[cell] = matching_neighbour(
                     leaf, x, y, z, neighbour_indices,
                     neighbour_values, active, values,
                     triangle_indices, false)
                     ? 1
                     : 0;
}

__global__ void transform_surface_values_kernel(
    const PackedSurfacePostGrid *grids,
    const int *leaf_grid_indices, size_t cell_count,
    const unsigned char *active, const double *values,
    double *output) {
  const size_t cell =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cell_count)
    return;
  const double value = values[cell];
  if (!active[cell]) {
    output[cell] = value;
    return;
  }
  const size_t leaf = cell / kLeafSize;
  const double voxel_size =
      grids[leaf_grid_indices[leaf]].voxel_size;
  const double weight = value < 0.0 ? voxel_size : -voxel_size;
  output[cell] = weight * sqrt(fabs(value));
}

struct Runtime {
  std::mutex mutex;
  cudaStream_t stream = nullptr;
  DeviceBuffer grids, leaf_grid_indices, leaf_origins;
  DeviceBuffer vertices, triangles;
  DeviceBuffer active, active_output;
  DeviceBuffer values, sign_values, validated_values;
  DeviceBuffer triangle_indices, neighbour_indices, neighbour_values;
  DeviceBuffer axis_neighbour_indices;
  DeviceBuffer changed_nodes_a, changed_nodes_b, changed_voxels;
  DeviceBuffer any_changed;
  PinnedBuffer host_grids, host_leaf_grid_indices, host_leaf_origins;
  PinnedBuffer host_vertices, host_triangles;
  PinnedBuffer host_active, host_values, host_triangle_indices;
  PinnedBuffer host_neighbour_indices, host_neighbour_values;
  PinnedBuffer host_axis_neighbour_indices;

  ~Runtime() {
    if (stream)
      cudaStreamDestroy(stream);
  }

  void ensure_stream() {
    if (!stream) {
      cuda_memory::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create sparse surface postprocess CUDA stream");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }
};

Runtime &runtime() {
  static Runtime state;
  return state;
}

} // namespace

void postprocess_sparse_surface_cuda(SparseSurfacePostGrid &grid) {
  std::vector<SparseSurfacePostGrid *> grids{&grid};
  postprocess_sparse_surface_cuda_batch(grids);
}

void postprocess_sparse_surface_cuda_batch(
    const std::vector<SparseSurfacePostGrid *> &grids) {
  size_t leaf_count = 0, vertex_count = 0, triangle_count = 0;
  for (SparseSurfacePostGrid *grid : grids) {
    if (!grid || !grid->mesh)
      throw std::invalid_argument(
          "sparse surface CUDA batch contains a null input");
    if (!std::isfinite(grid->scale) || grid->scale <= 0.0)
      throw std::invalid_argument(
          "sparse surface scale must be finite and positive");
    if (!std::isfinite(grid->voxel_size) ||
        grid->voxel_size <= 0.0)
      throw std::invalid_argument(
          "sparse surface voxel size must be finite and positive");
    if (grid->active.size() != grid->values.size() ||
        grid->active.size() != grid->triangle_indices.size() ||
        grid->active.size() % kLeafSize != 0)
      throw std::invalid_argument("sparse surface leaves are malformed");
    const size_t leaves = grid->active.size() / kLeafSize;
    if (grid->leaf_origins.size() != leaves * 3 ||
        grid->neighbour_indices.size() !=
            leaves * kLeafNeighbourCount ||
        grid->neighbour_values.size() !=
            leaves * kLeafNeighbourCount ||
        grid->axis_neighbour_indices.size() !=
            leaves * kAxisNeighbourCount)
      throw std::invalid_argument(
          "sparse surface topology is malformed");
    if (grid->mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        grid->mesh->triangles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        leaf_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                leaves ||
        vertex_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                grid->mesh->vertices.size() ||
        triangle_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                grid->mesh->triangles.size())
      throw std::overflow_error("sparse surface packed overflow");
    for (int neighbour : grid->neighbour_indices) {
      if (neighbour < -1 ||
          (neighbour >= 0 &&
           static_cast<size_t>(neighbour) >= leaves))
        throw std::invalid_argument(
            "sparse surface neighbour index is invalid");
    }
    for (int neighbour : grid->axis_neighbour_indices) {
      if (neighbour < -1 ||
          (neighbour >= 0 &&
           static_cast<size_t>(neighbour) >= leaves))
        throw std::invalid_argument(
            "sparse surface axis neighbour index is invalid");
    }
    for (size_t cell = 0; cell < grid->active.size(); ++cell) {
      if (grid->active[cell] &&
          (grid->triangle_indices[cell] < 0 ||
           static_cast<size_t>(grid->triangle_indices[cell]) >=
               grid->mesh->triangles.size()))
        throw std::invalid_argument(
            "sparse surface triangle index is invalid");
    }
    leaf_count += leaves;
    vertex_count += grid->mesh->vertices.size();
    triangle_count += grid->mesh->triangles.size();
  }
  if (grids.empty() || leaf_count == 0)
    return;
  const size_t cell_count = leaf_count * kLeafSize;
  constexpr size_t threads = 128;
  if ((cell_count + threads - 1) / threads >
      static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("sparse surface launch overflow");

  Runtime &state = runtime();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_stream();
#define ENSURE(buffer, bytes, message) state.buffer.ensure(bytes, message)
  ENSURE(grids, grids.size() * sizeof(PackedSurfacePostGrid),
         "allocate sparse surface grids");
  ENSURE(leaf_grid_indices, leaf_count * sizeof(int),
         "allocate sparse surface leaf owners");
  ENSURE(leaf_origins, leaf_count * sizeof(int3),
         "allocate sparse surface leaf origins");
  ENSURE(vertices, vertex_count * sizeof(float3),
         "allocate sparse surface vertices");
  ENSURE(triangles, triangle_count * sizeof(int3),
         "allocate sparse surface triangles");
  ENSURE(active, cell_count * sizeof(unsigned char),
         "allocate sparse surface activity");
  ENSURE(active_output, cell_count * sizeof(unsigned char),
         "allocate sparse surface output activity");
  ENSURE(values, cell_count * sizeof(double),
         "allocate sparse surface values");
  ENSURE(sign_values, cell_count * sizeof(double),
         "allocate sparse surface sign values");
  ENSURE(validated_values, cell_count * sizeof(double),
         "allocate sparse surface validated values");
  ENSURE(triangle_indices, cell_count * sizeof(int),
         "allocate sparse surface triangle indices");
  ENSURE(neighbour_indices,
         leaf_count * kLeafNeighbourCount * sizeof(int),
         "allocate sparse surface neighbours");
  ENSURE(neighbour_values,
         leaf_count * kLeafNeighbourCount * sizeof(double),
         "allocate sparse surface neighbour values");
  ENSURE(axis_neighbour_indices,
         leaf_count * kAxisNeighbourCount * sizeof(int),
         "allocate sparse surface axis neighbours");
  ENSURE(changed_nodes_a, leaf_count * sizeof(unsigned char),
         "allocate sparse surface changed nodes A");
  ENSURE(changed_nodes_b, leaf_count * sizeof(unsigned char),
         "allocate sparse surface changed nodes B");
  ENSURE(changed_voxels, cell_count * sizeof(unsigned char),
         "allocate sparse surface changed voxels");
  ENSURE(any_changed, sizeof(int),
         "allocate sparse surface change flag");
  ENSURE(host_grids, grids.size() * sizeof(PackedSurfacePostGrid),
         "allocate host sparse surface grids");
  ENSURE(host_leaf_grid_indices, leaf_count * sizeof(int),
         "allocate host sparse surface leaf owners");
  ENSURE(host_leaf_origins, leaf_count * sizeof(int3),
         "allocate host sparse surface leaf origins");
  ENSURE(host_vertices, vertex_count * sizeof(float3),
         "allocate host sparse surface vertices");
  ENSURE(host_triangles, triangle_count * sizeof(int3),
         "allocate host sparse surface triangles");
  ENSURE(host_active, cell_count * sizeof(unsigned char),
         "allocate host sparse surface activity");
  ENSURE(host_values, cell_count * sizeof(double),
         "allocate host sparse surface values");
  ENSURE(host_triangle_indices, cell_count * sizeof(int),
         "allocate host sparse surface triangle indices");
  ENSURE(host_neighbour_indices,
         leaf_count * kLeafNeighbourCount * sizeof(int),
         "allocate host sparse surface neighbours");
  ENSURE(host_neighbour_values,
         leaf_count * kLeafNeighbourCount * sizeof(double),
         "allocate host sparse surface neighbour values");
  ENSURE(host_axis_neighbour_indices,
         leaf_count * kAxisNeighbourCount * sizeof(int),
         "allocate host sparse surface axis neighbours");
#undef ENSURE

  auto *host_grids = state.host_grids.as<PackedSurfacePostGrid>();
  int *host_owners = state.host_leaf_grid_indices.as<int>();
  int3 *host_origins = state.host_leaf_origins.as<int3>();
  float3 *host_vertices = state.host_vertices.as<float3>();
  int3 *host_triangles = state.host_triangles.as<int3>();
  unsigned char *host_active =
      state.host_active.as<unsigned char>();
  double *host_values = state.host_values.as<double>();
  int *host_indices = state.host_triangle_indices.as<int>();
  int *host_neighbours = state.host_neighbour_indices.as<int>();
  double *host_neighbour_values =
      state.host_neighbour_values.as<double>();
  int *host_axis_neighbours =
      state.host_axis_neighbour_indices.as<int>();
  size_t leaf_offset = 0, vertex_offset = 0, triangle_offset = 0;
  for (size_t grid_index = 0; grid_index < grids.size(); ++grid_index) {
    SparseSurfacePostGrid &grid = *grids[grid_index];
    const size_t leaves = grid.active.size() / kLeafSize;
    host_grids[grid_index] =
        {triangle_offset, grid.voxel_size};
    std::fill_n(host_owners + leaf_offset, leaves,
                static_cast<int>(grid_index));
    for (size_t leaf = 0; leaf < leaves; ++leaf) {
      host_origins[leaf_offset + leaf] = make_int3(
          grid.leaf_origins[leaf * 3],
          grid.leaf_origins[leaf * 3 + 1],
          grid.leaf_origins[leaf * 3 + 2]);
    }
    const Mesh &mesh = *grid.mesh;
    for (size_t vertex = 0; vertex < mesh.vertices.size(); ++vertex) {
      const Vec3D &point = mesh.vertices[vertex];
      host_vertices[vertex_offset + vertex] = make_float3(
          static_cast<float>(point[0] * grid.scale),
          static_cast<float>(point[1] * grid.scale),
          static_cast<float>(point[2] * grid.scale));
    }
    for (size_t triangle = 0; triangle < mesh.triangles.size();
         ++triangle) {
      const auto &face = mesh.triangles[triangle];
      host_triangles[triangle_offset + triangle] = make_int3(
          static_cast<int>(vertex_offset) + face[0],
          static_cast<int>(vertex_offset) + face[1],
          static_cast<int>(vertex_offset) + face[2]);
    }
    std::copy(grid.active.begin(), grid.active.end(),
              host_active + leaf_offset * kLeafSize);
    std::copy(grid.values.begin(), grid.values.end(),
              host_values + leaf_offset * kLeafSize);
    std::copy(grid.triangle_indices.begin(),
              grid.triangle_indices.end(),
              host_indices + leaf_offset * kLeafSize);
    for (size_t index = 0;
         index < leaves * kLeafNeighbourCount; ++index) {
      const int local = grid.neighbour_indices[index];
      host_neighbours[leaf_offset * kLeafNeighbourCount + index] =
          local < 0 ? -1 : static_cast<int>(leaf_offset) + local;
      host_neighbour_values[
          leaf_offset * kLeafNeighbourCount + index] =
          grid.neighbour_values[index];
    }
    for (size_t index = 0;
         index < leaves * kAxisNeighbourCount; ++index) {
      const int local = grid.axis_neighbour_indices[index];
      host_axis_neighbours[
          leaf_offset * kAxisNeighbourCount + index] =
          local < 0 ? -1 : static_cast<int>(leaf_offset) + local;
    }
    leaf_offset += leaves;
    vertex_offset += mesh.vertices.size();
    triangle_offset += mesh.triangles.size();
  }

  const auto copy_to_device = [&](void *destination, const void *source,
                                  size_t bytes, const char *message) {
    cuda_memory::check(
        cudaMemcpyAsync(destination, source, bytes,
                        cudaMemcpyHostToDevice, state.stream),
        message);
  };
#define COPY(buffer, type, source, bytes, message) \
  copy_to_device(state.buffer.as<type>(), source, bytes, message)
  COPY(grids, PackedSurfacePostGrid, host_grids,
       grids.size() * sizeof(PackedSurfacePostGrid),
       "copy sparse surface grids");
  COPY(leaf_grid_indices, int, host_owners, leaf_count * sizeof(int),
       "copy sparse surface leaf owners");
  COPY(leaf_origins, int3, host_origins, leaf_count * sizeof(int3),
       "copy sparse surface leaf origins");
  COPY(vertices, float3, host_vertices, vertex_count * sizeof(float3),
       "copy sparse surface vertices");
  COPY(triangles, int3, host_triangles, triangle_count * sizeof(int3),
       "copy sparse surface triangles");
  COPY(active, unsigned char, host_active,
       cell_count * sizeof(unsigned char),
       "copy sparse surface activity");
  COPY(values, double, host_values, cell_count * sizeof(double),
       "copy sparse surface values");
  COPY(triangle_indices, int, host_indices, cell_count * sizeof(int),
       "copy sparse surface triangle indices");
  COPY(neighbour_indices, int, host_neighbours,
       leaf_count * kLeafNeighbourCount * sizeof(int),
       "copy sparse surface neighbours");
  COPY(neighbour_values, double, host_neighbour_values,
       leaf_count * kLeafNeighbourCount * sizeof(double),
       "copy sparse surface neighbour values");
  COPY(axis_neighbour_indices, int, host_axis_neighbours,
       leaf_count * kAxisNeighbourCount * sizeof(int),
       "copy sparse surface axis neighbours");
#undef COPY

  const int blocks =
      static_cast<int>((cell_count + threads - 1) / threads);
  const int leaf_blocks =
      static_cast<int>((leaf_count + threads - 1) / threads);
  const size_t line_count = leaf_count * 64;
  const int line_blocks =
      static_cast<int>((line_count + threads - 1) / threads);
  for (int axis = 2; axis >= 0; --axis) {
    sweep_exterior_kernel<<<line_blocks, threads, 0, state.stream>>>(
        leaf_count, axis,
        state.axis_neighbour_indices.as<int>(),
        state.values.as<double>());
    cuda_memory::check(cudaGetLastError(),
                       "launch sparse exterior sweep");
  }
  cuda_memory::check(
      cudaMemsetAsync(state.changed_nodes_a.as<unsigned char>(), 1,
                      leaf_count * sizeof(unsigned char), state.stream),
      "initialize sparse exterior changed nodes A");
  cuda_memory::check(
      cudaMemsetAsync(state.changed_nodes_b.as<unsigned char>(), 0,
                      leaf_count * sizeof(unsigned char), state.stream),
      "initialize sparse exterior changed nodes B");
  cuda_memory::check(
      cudaMemsetAsync(state.changed_voxels.as<unsigned char>(), 0,
                      cell_count * sizeof(unsigned char), state.stream),
      "initialize sparse exterior changed voxels");
  unsigned char *changed_nodes =
      state.changed_nodes_a.as<unsigned char>();
  unsigned char *next_changed_nodes =
      state.changed_nodes_b.as<unsigned char>();
  while (true) {
    scan_fill_exterior_kernel<<<leaf_blocks, threads, 0,
                                state.stream>>>(
        leaf_count, changed_nodes, state.values.as<double>());
    cuda_memory::check(cudaGetLastError(),
                       "launch sparse exterior scan fill");
    cuda_memory::check(
        cudaMemsetAsync(state.any_changed.as<int>(), 0, sizeof(int),
                        state.stream),
        "clear sparse exterior change flag");
    seed_exterior_points_kernel<<<leaf_blocks, threads, 0,
                                  state.stream>>>(
        leaf_count, state.axis_neighbour_indices.as<int>(),
        changed_nodes, next_changed_nodes,
        state.values.as<double>(),
        state.changed_voxels.as<unsigned char>(),
        state.any_changed.as<int>());
    cuda_memory::check(cudaGetLastError(),
                       "launch sparse exterior seed points");
    int any_changed = 0;
    cuda_memory::check(
        cudaMemcpyAsync(&any_changed, state.any_changed.as<int>(),
                        sizeof(int), cudaMemcpyDeviceToHost,
                        state.stream),
        "copy sparse exterior change flag");
    cuda_memory::check(cudaStreamSynchronize(state.stream),
                       "wait for sparse exterior iteration");
    std::swap(changed_nodes, next_changed_nodes);
    if (!any_changed)
      break;
    sync_exterior_voxels_kernel<<<leaf_blocks, threads, 0,
                                  state.stream>>>(
        leaf_count, changed_nodes,
        state.changed_voxels.as<unsigned char>(),
        state.values.as<double>());
    cuda_memory::check(cudaGetLastError(),
                       "launch sparse exterior voxel sync");
  }
  compute_surface_sign_kernel<<<blocks, threads, 0, state.stream>>>(
      state.grids.as<PackedSurfacePostGrid>(),
      state.leaf_grid_indices.as<int>(),
      state.leaf_origins.as<int3>(), cell_count,
      state.neighbour_indices.as<int>(),
      state.neighbour_values.as<double>(),
      state.active.as<unsigned char>(), state.values.as<double>(),
      state.triangle_indices.as<int>(), state.vertices.as<float3>(),
      state.triangles.as<int3>(), state.sign_values.as<double>());
  cuda_memory::check(cudaGetLastError(),
                     "launch sparse surface sign");
  validate_surface_kernel<<<blocks, threads, 0, state.stream>>>(
      state.leaf_grid_indices.as<int>(), cell_count,
      state.neighbour_indices.as<int>(),
      state.neighbour_values.as<double>(),
      state.active.as<unsigned char>(),
      state.sign_values.as<double>(),
      state.triangle_indices.as<int>(),
      state.validated_values.as<double>());
  cuda_memory::check(cudaGetLastError(),
                     "launch sparse surface validation");
  cleanup_surface_kernel<<<blocks, threads, 0, state.stream>>>(
      cell_count, state.neighbour_indices.as<int>(),
      state.neighbour_values.as<double>(),
      state.active.as<unsigned char>(),
      state.validated_values.as<double>(),
      state.triangle_indices.as<int>(),
      state.active_output.as<unsigned char>());
  cuda_memory::check(cudaGetLastError(),
                     "launch sparse surface cleanup");
  transform_surface_values_kernel<<<blocks, threads, 0, state.stream>>>(
      state.grids.as<PackedSurfacePostGrid>(),
      state.leaf_grid_indices.as<int>(), cell_count,
      state.active_output.as<unsigned char>(),
      state.validated_values.as<double>(),
      state.sign_values.as<double>());
  cuda_memory::check(cudaGetLastError(),
                     "launch sparse surface value transformation");
  cuda_memory::check(
      cudaMemcpyAsync(host_values, state.sign_values.as<double>(),
                      cell_count * sizeof(double),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy sparse surface values");
  cuda_memory::check(
      cudaMemcpyAsync(host_active,
                      state.active_output.as<unsigned char>(),
                      cell_count * sizeof(unsigned char),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy sparse surface activity");
  cuda_memory::check(cudaStreamSynchronize(state.stream),
                     "wait for sparse surface postprocess");

  leaf_offset = 0;
  for (SparseSurfacePostGrid *grid : grids) {
    std::copy_n(host_values + leaf_offset * kLeafSize,
                grid->values.size(), grid->values.begin());
    std::copy_n(host_active + leaf_offset * kLeafSize,
                grid->active.size(), grid->active.begin());
    leaf_offset += grid->active.size() / kLeafSize;
  }
}

} // namespace neural_acd
