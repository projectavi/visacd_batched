#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <preprocess_cuda.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace neural_acd {
namespace {

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

constexpr size_t kMaximumAutomaticWave = 200;

struct PackedGrid {
  int3 minimum;
  int3 dimensions;
  size_t cells = 0;
  size_t cell_offset = 0;
  size_t triangle_offset = 0;
};

struct PackedSeedVoxel {
  int3 coordinate;
  int global_triangle_index;
  int mesh_index;
};

struct PreparedSurface {
  size_t input_index = 0;
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  size_t candidate_voxels = 0;
  PackedGrid grid;
};

struct PackedCounts {
  size_t meshes = 0;
  size_t vertices = 0;
  size_t triangles = 0;
  size_t cells = 0;
  size_t candidates = 0;
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

__device__ double squared_distance(double3 a, double3 b, double3 c,
                                   int x, int y, int z) {
  const double3 point = make_double3(static_cast<double>(x),
                                     static_cast<double>(y),
                                     static_cast<double>(z));
  const double3 closest = closest_triangle(a, c, b, point);
  const double3 delta = subtract(point, closest);
  return dot_product(delta, delta);
}

__device__ size_t dense_offset(int x, int y, int z,
                               const PackedGrid &grid) {
  return grid.cell_offset +
         (static_cast<size_t>(x - grid.minimum.x) * grid.dimensions.y +
          static_cast<size_t>(y - grid.minimum.y)) *
             grid.dimensions.z +
         static_cast<size_t>(z - grid.minimum.z);
}

__global__ void find_seed_voxels_kernel(
    const float3 *vertices, const int3 *triangles,
    const int *triangle_mesh_indices, size_t triangle_count,
    PackedSeedVoxel *seeds, unsigned int *seed_count,
    size_t seed_capacity) {
  const size_t triangle_index = blockIdx.x;
  if (triangle_index >= triangle_count)
    return;
  const int3 triangle = triangles[triangle_index];
  const float3 af = vertices[triangle.x];
  const float3 bf = vertices[triangle.y];
  const float3 cf = vertices[triangle.z];
  const double3 a = make_double3(af.x, af.y, af.z);
  const double3 b = make_double3(bf.x, bf.y, bf.z);
  const double3 c = make_double3(cf.x, cf.y, cf.z);

  const int minimum_x = __double2int_rd(fmin(a.x, fmin(b.x, c.x))) - 1;
  const int minimum_y = __double2int_rd(fmin(a.y, fmin(b.y, c.y))) - 1;
  const int minimum_z = __double2int_rd(fmin(a.z, fmin(b.z, c.z))) - 1;
  const int maximum_x = __double2int_ru(fmax(a.x, fmax(b.x, c.x))) + 1;
  const int maximum_y = __double2int_ru(fmax(a.y, fmax(b.y, c.y))) + 1;
  const int maximum_z = __double2int_ru(fmax(a.z, fmax(b.z, c.z))) + 1;
  const size_t size_y = static_cast<size_t>(maximum_y - minimum_y + 1);
  const size_t size_z = static_cast<size_t>(maximum_z - minimum_z + 1);
  const size_t count =
      static_cast<size_t>(maximum_x - minimum_x + 1) * size_y * size_z;

  for (size_t local = threadIdx.x; local < count; local += blockDim.x) {
    const int x = minimum_x + static_cast<int>(local / (size_y * size_z));
    const size_t remainder = local % (size_y * size_z);
    const int y = minimum_y + static_cast<int>(remainder / size_z);
    const int z = minimum_z + static_cast<int>(remainder % size_z);
    const double distance = squared_distance(a, b, c, x, y, z);
    if (distance > 0.75)
      continue;
    const unsigned int output = atomicAdd(seed_count, 1u);
    if (output < seed_capacity) {
      seeds[output] = {make_int3(x, y, z),
                       static_cast<int>(triangle_index),
                       triangle_mesh_indices[triangle_index]};
    }
  }
}

template <bool StoreTriangles>
__device__ void store_voxel(double3 a, double3 b, double3 c, int x, int y,
                            int z, int triangle_index,
                            const PackedGrid &grid,
                            unsigned long long *distance_bits,
                            int *triangle_indices) {
  const double distance = squared_distance(a, b, c, x, y, z);
  const size_t offset = dense_offset(x, y, z, grid);
  const unsigned long long bits = __double_as_longlong(distance);
  if constexpr (StoreTriangles) {
    if (distance_bits[offset] == bits)
      atomicMin(triangle_indices + offset, triangle_index);
  } else {
    atomicMin(distance_bits + offset, bits);
  }
}

template <bool StoreTriangles>
__global__ void expand_seed_voxels_kernel(
    const float3 *vertices, const int3 *triangles,
    const PackedSeedVoxel *seeds, const unsigned int *seed_count,
    const PackedGrid *grids, unsigned long long *distance_bits,
    int *triangle_indices) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t seed_index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       seed_index < *seed_count; seed_index += stride) {
    const PackedSeedVoxel seed = seeds[seed_index];
    const PackedGrid grid = grids[seed.mesh_index];
    const int3 triangle = triangles[seed.global_triangle_index];
    const float3 af = vertices[triangle.x];
    const float3 bf = vertices[triangle.y];
    const float3 cf = vertices[triangle.z];
    const double3 a = make_double3(af.x, af.y, af.z);
    const double3 b = make_double3(bf.x, bf.y, bf.z);
    const double3 c = make_double3(cf.x, cf.y, cf.z);
    const int local_triangle =
        seed.global_triangle_index - static_cast<int>(grid.triangle_offset);
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          store_voxel<StoreTriangles>(
              a, b, c, seed.coordinate.x + dx, seed.coordinate.y + dy,
              seed.coordinate.z + dz, local_triangle, grid,
              distance_bits, triangle_indices);
        }
      }
    }
  }
}

template <bool StoreTriangles>
__global__ void store_initial_neighbourhood_kernel(
    const float3 *vertices, const int3 *triangles,
    const int *triangle_mesh_indices, size_t triangle_count,
    const PackedGrid *grids, unsigned long long *distance_bits,
    int *triangle_indices) {
  const size_t triangle_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (triangle_index >= triangle_count)
    return;
  const PackedGrid grid = grids[triangle_mesh_indices[triangle_index]];
  const int3 triangle = triangles[triangle_index];
  const float3 af = vertices[triangle.x];
  const float3 bf = vertices[triangle.y];
  const float3 cf = vertices[triangle.z];
  const double3 a = make_double3(af.x, af.y, af.z);
  const double3 b = make_double3(bf.x, bf.y, bf.z);
  const double3 c = make_double3(cf.x, cf.y, cf.z);
  const int ax = __double2int_rd(a.x);
  const int ay = __double2int_rd(a.y);
  const int az = __double2int_rd(a.z);
  const int local_triangle =
      static_cast<int>(triangle_index - grid.triangle_offset);
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        store_voxel<StoreTriangles>(
            a, b, c, ax + dx, ay + dy, az + dz, local_triangle, grid,
            distance_bits, triangle_indices);
      }
    }
  }
}

bool checked_add(size_t first, size_t second, size_t &result) {
  if (second > std::numeric_limits<size_t>::max() - first)
    return false;
  result = first + second;
  return true;
}

bool checked_multiply(size_t first, size_t second, size_t &result) {
  if (first != 0 && second > std::numeric_limits<size_t>::max() / first)
    return false;
  result = first * second;
  return true;
}

bool finite_vertex(float3 vertex) {
  return std::isfinite(vertex.x) && std::isfinite(vertex.y) &&
         std::isfinite(vertex.z);
}

bool same_vertex(float3 first, float3 second) {
  return first.x == second.x && first.y == second.y &&
         first.z == second.z;
}

void set_fallback(SurfaceVoxelizationResult &result,
                  const char *reason) {
  result.supported = false;
  result.fallback_reason = reason;
  result.records.clear();
}

bool prepare_surface(const SurfaceVoxelizationInput &input,
                     size_t input_index, PreparedSurface &prepared) {
  SurfaceVoxelizationResult &result = *input.result;
  const Mesh &mesh = *input.mesh;
  if (!std::isfinite(input.scale) || input.scale <= 0.0) {
    set_fallback(result, "scale must be finite and positive");
    return false;
  }
  if (mesh.vertices.empty() || mesh.triangles.empty()) {
    set_fallback(result, "mesh is empty");
    return false;
  }
  if (mesh.vertices.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      mesh.triangles.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    set_fallback(result, "mesh index range exceeds CUDA limits");
    return false;
  }

  std::vector<float3> vertices(mesh.vertices.size());
  float3 minimum = make_float3(std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::infinity());
  float3 maximum = make_float3(-std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity());
  for (size_t index = 0; index < mesh.vertices.size(); ++index) {
    const Vec3D &source = mesh.vertices[index];
    const float3 vertex =
        make_float3(static_cast<float>(source[0] * input.scale),
                    static_cast<float>(source[1] * input.scale),
                    static_cast<float>(source[2] * input.scale));
    if (!finite_vertex(vertex)) {
      set_fallback(result, "mesh contains non-finite vertices");
      return false;
    }
    vertices[index] = vertex;
    minimum.x = std::min(minimum.x, vertex.x);
    minimum.y = std::min(minimum.y, vertex.y);
    minimum.z = std::min(minimum.z, vertex.z);
    maximum.x = std::max(maximum.x, vertex.x);
    maximum.y = std::max(maximum.y, vertex.y);
    maximum.z = std::max(maximum.z, vertex.z);
  }

  constexpr float coordinate_minimum =
      static_cast<float>(std::numeric_limits<int>::min() + 4);
  constexpr float coordinate_maximum =
      static_cast<float>(std::numeric_limits<int>::max() - 4);
  if (minimum.x < coordinate_minimum || minimum.y < coordinate_minimum ||
      minimum.z < coordinate_minimum || maximum.x > coordinate_maximum ||
      maximum.y > coordinate_maximum || maximum.z > coordinate_maximum) {
    set_fallback(result, "scaled mesh exceeds CUDA coordinate limits");
    return false;
  }

  size_t candidate_voxels = 0;
  for (const auto &source : mesh.triangles) {
    if (source[0] < 0 || source[1] < 0 || source[2] < 0 ||
        static_cast<size_t>(source[0]) >= mesh.vertices.size() ||
        static_cast<size_t>(source[1]) >= mesh.vertices.size() ||
        static_cast<size_t>(source[2]) >= mesh.vertices.size()) {
      set_fallback(result, "mesh contains invalid triangle indices");
      return false;
    }
    const float3 a = vertices[source[0]];
    const float3 b = vertices[source[1]];
    const float3 c = vertices[source[2]];
    if (same_vertex(a, b) || same_vertex(a, c) || same_vertex(b, c)) {
      set_fallback(result, "mesh contains duplicate float vertices");
      return false;
    }
    const size_t x = static_cast<size_t>(
        static_cast<long long>(std::ceil(std::max({a.x, b.x, c.x}))) -
        static_cast<long long>(std::floor(std::min({a.x, b.x, c.x}))) + 3);
    const size_t y = static_cast<size_t>(
        static_cast<long long>(std::ceil(std::max({a.y, b.y, c.y}))) -
        static_cast<long long>(std::floor(std::min({a.y, b.y, c.y}))) + 3);
    const size_t z = static_cast<size_t>(
        static_cast<long long>(std::ceil(std::max({a.z, b.z, c.z}))) -
        static_cast<long long>(std::floor(std::min({a.z, b.z, c.z}))) + 3);
    size_t xy = 0;
    size_t xyz = 0;
    size_t next = 0;
    if (!checked_multiply(x, y, xy) || !checked_multiply(xy, z, xyz) ||
        !checked_add(candidate_voxels, xyz, next)) {
      set_fallback(result, "voxel candidate count overflows");
      return false;
    }
    candidate_voxels = next;
  }
  result.candidate_voxels = candidate_voxels;
  if (candidate_voxels > std::numeric_limits<unsigned int>::max()) {
    set_fallback(result, "seed voxel count exceeds CUDA limits");
    return false;
  }

  PackedGrid grid;
  grid.minimum = make_int3(static_cast<int>(std::floor(minimum.x)) - 3,
                           static_cast<int>(std::floor(minimum.y)) - 3,
                           static_cast<int>(std::floor(minimum.z)) - 3);
  const int3 grid_maximum =
      make_int3(static_cast<int>(std::ceil(maximum.x)) + 3,
                static_cast<int>(std::ceil(maximum.y)) + 3,
                static_cast<int>(std::ceil(maximum.z)) + 3);
  grid.dimensions = make_int3(grid_maximum.x - grid.minimum.x + 1,
                              grid_maximum.y - grid.minimum.y + 1,
                              grid_maximum.z - grid.minimum.z + 1);
  size_t xy_cells = 0;
  if (grid.dimensions.x <= 0 || grid.dimensions.y <= 0 ||
      grid.dimensions.z <= 0 ||
      !checked_multiply(static_cast<size_t>(grid.dimensions.x),
                        static_cast<size_t>(grid.dimensions.y), xy_cells) ||
      !checked_multiply(xy_cells, static_cast<size_t>(grid.dimensions.z),
                        grid.cells)) {
    set_fallback(result, "dense voxel bounds overflow");
    return false;
  }

  prepared.input_index = input_index;
  prepared.vertex_count = mesh.vertices.size();
  prepared.triangle_count = mesh.triangles.size();
  prepared.candidate_voxels = candidate_voxels;
  prepared.grid = grid;
  return true;
}

bool append_counts(const PreparedSurface &surface,
                   const PackedCounts &current, PackedCounts &next) {
  next = current;
  if (current.meshes >=
      static_cast<size_t>(std::numeric_limits<int>::max()))
    return false;
  ++next.meshes;
  if (!checked_add(current.vertices, surface.vertex_count, next.vertices) ||
      next.vertices > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      !checked_add(current.triangles, surface.triangle_count,
                   next.triangles) ||
      next.triangles >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      !checked_add(current.cells, surface.grid.cells, next.cells) ||
      !checked_add(current.candidates, surface.candidate_voxels,
                   next.candidates) ||
      next.candidates > std::numeric_limits<unsigned int>::max()) {
    return false;
  }
  return true;
}

size_t packed_bytes(const PackedCounts &counts) {
  size_t total = 0;
  const auto include = [&](size_t count, size_t element_size) {
    size_t bytes = 0;
    size_t next = 0;
    if (!checked_multiply(count, element_size, bytes) ||
        !checked_add(total, bytes, next)) {
      total = std::numeric_limits<size_t>::max();
      return;
    }
    total = next;
  };
  include(counts.vertices, sizeof(float3));
  include(counts.triangles, sizeof(int3));
  include(counts.triangles, sizeof(int));
  include(counts.meshes, sizeof(PackedGrid));
  include(counts.cells, sizeof(unsigned long long));
  include(counts.cells, sizeof(int));
  include(counts.candidates, sizeof(PackedSeedVoxel));
  include(1, sizeof(unsigned int));
  return total;
}

} // namespace

struct ManifoldCudaBatchRuntime::Impl {
  cudaStream_t stream = nullptr;
  cudaEvent_t started = nullptr;
  cudaEvent_t finished = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer triangles;
  DeviceBuffer triangle_mesh_indices;
  DeviceBuffer grids;
  DeviceBuffer distance_bits;
  DeviceBuffer triangle_indices;
  DeviceBuffer seeds;
  DeviceBuffer seed_count;
  PinnedBuffer host_vertices;
  PinnedBuffer host_triangles;
  PinnedBuffer host_triangle_mesh_indices;
  PinnedBuffer host_grids;
  PinnedBuffer host_distance_bits;
  PinnedBuffer host_triangle_indices;

  ~Impl() {
    if (stream)
      cudaStreamSynchronize(stream);
    if (started)
      cudaEventDestroy(started);
    if (finished)
      cudaEventDestroy(finished);
    if (stream)
      cudaStreamDestroy(stream);
  }

  void ensure_runtime() {
    if (!stream) {
      cuda_memory::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create batched manifold preprocessing stream");
      cuda_memory::check(cudaEventCreate(&started),
                         "create batched manifold start event");
      cuda_memory::check(cudaEventCreate(&finished),
                         "create batched manifold finish event");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }
};

ManifoldCudaBatchRuntime::ManifoldCudaBatchRuntime()
    : impl_(std::make_unique<Impl>()) {}
ManifoldCudaBatchRuntime::~ManifoldCudaBatchRuntime() = default;
ManifoldCudaBatchRuntime::ManifoldCudaBatchRuntime(
    ManifoldCudaBatchRuntime &&) noexcept = default;
ManifoldCudaBatchRuntime &ManifoldCudaBatchRuntime::operator=(
    ManifoldCudaBatchRuntime &&) noexcept = default;

namespace {

void run_wave(const std::vector<SurfaceVoxelizationInput> &inputs,
              const std::vector<PreparedSurface> &prepared, size_t begin,
              size_t end, const PackedCounts &counts,
              ManifoldCudaBatchRuntime::Impl &runtime) {
  runtime.ensure_runtime();
  runtime.vertices.ensure(counts.vertices * sizeof(float3),
                          "allocate batched manifold vertices");
  runtime.triangles.ensure(counts.triangles * sizeof(int3),
                           "allocate batched manifold triangles");
  runtime.triangle_mesh_indices.ensure(
      counts.triangles * sizeof(int),
      "allocate batched manifold triangle owners");
  runtime.grids.ensure(counts.meshes * sizeof(PackedGrid),
                       "allocate batched manifold grids");
  runtime.distance_bits.ensure(
      counts.cells * sizeof(unsigned long long),
      "allocate batched manifold distance grids");
  runtime.triangle_indices.ensure(
      counts.cells * sizeof(int),
      "allocate batched manifold triangle grids");
  runtime.seeds.ensure(counts.candidates * sizeof(PackedSeedVoxel),
                       "allocate batched manifold seeds");
  runtime.seed_count.ensure(sizeof(unsigned int),
                            "allocate batched manifold seed count");
  runtime.host_vertices.ensure(counts.vertices * sizeof(float3),
                               "allocate host batched manifold vertices");
  runtime.host_triangles.ensure(
      counts.triangles * sizeof(int3),
      "allocate host batched manifold triangles");
  runtime.host_triangle_mesh_indices.ensure(
      counts.triangles * sizeof(int),
      "allocate host batched manifold triangle owners");
  runtime.host_grids.ensure(counts.meshes * sizeof(PackedGrid),
                            "allocate host batched manifold grids");
  runtime.host_distance_bits.ensure(
      counts.cells * sizeof(unsigned long long),
      "allocate host batched manifold distances");
  runtime.host_triangle_indices.ensure(
      counts.cells * sizeof(int),
      "allocate host batched manifold triangle indices");

  float3 *host_vertices = runtime.host_vertices.as<float3>();
  int3 *host_triangles = runtime.host_triangles.as<int3>();
  int *host_triangle_mesh_indices =
      runtime.host_triangle_mesh_indices.as<int>();
  PackedGrid *host_grids = runtime.host_grids.as<PackedGrid>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  size_t cell_offset = 0;
  for (size_t relative = 0; relative < end - begin; ++relative) {
    const PreparedSurface &surface = prepared[begin + relative];
    const SurfaceVoxelizationInput &input = inputs[surface.input_index];
    const Mesh &mesh = *input.mesh;
    PackedGrid grid = surface.grid;
    grid.cell_offset = cell_offset;
    grid.triangle_offset = triangle_offset;
    host_grids[relative] = grid;
    for (const Vec3D &source : mesh.vertices) {
      host_vertices[vertex_offset++] =
          make_float3(static_cast<float>(source[0] * input.scale),
                      static_cast<float>(source[1] * input.scale),
                      static_cast<float>(source[2] * input.scale));
    }
    const size_t mesh_vertex_offset = vertex_offset - mesh.vertices.size();
    for (const auto &triangle : mesh.triangles) {
      host_triangles[triangle_offset] = make_int3(
          static_cast<int>(mesh_vertex_offset) + triangle[0],
          static_cast<int>(mesh_vertex_offset) + triangle[1],
          static_cast<int>(mesh_vertex_offset) + triangle[2]);
      host_triangle_mesh_indices[triangle_offset] =
          static_cast<int>(relative);
      ++triangle_offset;
    }
    cell_offset += grid.cells;
  }

  cuda_memory::check(cudaEventRecord(runtime.started, runtime.stream),
                     "record batched manifold start");
  cuda_memory::check(
      cudaMemcpyAsync(runtime.vertices.as<float3>(), host_vertices,
                      counts.vertices * sizeof(float3),
                      cudaMemcpyHostToDevice, runtime.stream),
      "copy batched manifold vertices");
  cuda_memory::check(
      cudaMemcpyAsync(runtime.triangles.as<int3>(), host_triangles,
                      counts.triangles * sizeof(int3),
                      cudaMemcpyHostToDevice, runtime.stream),
      "copy batched manifold triangles");
  cuda_memory::check(
      cudaMemcpyAsync(runtime.triangle_mesh_indices.as<int>(),
                      host_triangle_mesh_indices,
                      counts.triangles * sizeof(int),
                      cudaMemcpyHostToDevice, runtime.stream),
      "copy batched manifold triangle owners");
  cuda_memory::check(
      cudaMemcpyAsync(runtime.grids.as<PackedGrid>(), host_grids,
                      counts.meshes * sizeof(PackedGrid),
                      cudaMemcpyHostToDevice, runtime.stream),
      "copy batched manifold grids");
  cuda_memory::check(
      cudaMemsetAsync(runtime.distance_bits.as<unsigned long long>(), 0xff,
                      counts.cells * sizeof(unsigned long long),
                      runtime.stream),
      "initialize batched manifold distances");
  cuda_memory::check(
      cudaMemsetAsync(runtime.triangle_indices.as<int>(), 0x7f,
                      counts.cells * sizeof(int), runtime.stream),
      "initialize batched manifold triangles");
  cuda_memory::check(
      cudaMemsetAsync(runtime.seed_count.as<unsigned int>(), 0,
                      sizeof(unsigned int), runtime.stream),
      "initialize batched manifold seed count");

  constexpr int threads = 128;
  find_seed_voxels_kernel<<<counts.triangles, threads, 0, runtime.stream>>>(
      runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
      runtime.triangle_mesh_indices.as<int>(), counts.triangles,
      runtime.seeds.as<PackedSeedVoxel>(),
      runtime.seed_count.as<unsigned int>(), counts.candidates);
  cuda_memory::check(cudaGetLastError(),
                     "launch batched manifold seed voxelization");
  const int triangle_blocks = static_cast<int>(
      (counts.triangles + threads - 1) / threads);
  const int seed_blocks = static_cast<int>(std::max<size_t>(
      1, std::min<size_t>(4096,
                          (counts.candidates + threads - 1) / threads)));
  store_initial_neighbourhood_kernel<false>
      <<<triangle_blocks, threads, 0, runtime.stream>>>(
          runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
          runtime.triangle_mesh_indices.as<int>(), counts.triangles,
          runtime.grids.as<PackedGrid>(),
          runtime.distance_bits.as<unsigned long long>(),
          runtime.triangle_indices.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "launch batched manifold initial distance halo");
  expand_seed_voxels_kernel<false>
      <<<seed_blocks, threads, 0, runtime.stream>>>(
          runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
          runtime.seeds.as<PackedSeedVoxel>(),
          runtime.seed_count.as<unsigned int>(),
          runtime.grids.as<PackedGrid>(),
          runtime.distance_bits.as<unsigned long long>(),
          runtime.triangle_indices.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "launch batched manifold distance voxelization");
  store_initial_neighbourhood_kernel<true>
      <<<triangle_blocks, threads, 0, runtime.stream>>>(
          runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
          runtime.triangle_mesh_indices.as<int>(), counts.triangles,
          runtime.grids.as<PackedGrid>(),
          runtime.distance_bits.as<unsigned long long>(),
          runtime.triangle_indices.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "launch batched manifold initial triangle halo");
  expand_seed_voxels_kernel<true>
      <<<seed_blocks, threads, 0, runtime.stream>>>(
          runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
          runtime.seeds.as<PackedSeedVoxel>(),
          runtime.seed_count.as<unsigned int>(),
          runtime.grids.as<PackedGrid>(),
          runtime.distance_bits.as<unsigned long long>(),
          runtime.triangle_indices.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "launch batched manifold triangle voxelization");

  cuda_memory::check(
      cudaMemcpyAsync(runtime.host_distance_bits.as<unsigned long long>(),
                      runtime.distance_bits.as<unsigned long long>(),
                      counts.cells * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost, runtime.stream),
      "copy batched manifold distances");
  cuda_memory::check(
      cudaMemcpyAsync(runtime.host_triangle_indices.as<int>(),
                      runtime.triangle_indices.as<int>(),
                      counts.cells * sizeof(int), cudaMemcpyDeviceToHost,
                      runtime.stream),
      "copy batched manifold triangle indices");
  cuda_memory::check(cudaEventRecord(runtime.finished, runtime.stream),
                     "record batched manifold finish");
  cuda_memory::check(cudaEventSynchronize(runtime.finished),
                     "wait for batched manifold preprocessing");
  float milliseconds = 0.0f;
  cuda_memory::check(
      cudaEventElapsedTime(&milliseconds, runtime.started, runtime.finished),
      "time batched manifold preprocessing");

  const unsigned long long *host_distances =
      runtime.host_distance_bits.as<unsigned long long>();
  const int *host_indices = runtime.host_triangle_indices.as<int>();
  for (size_t relative = 0; relative < end - begin; ++relative) {
    const PreparedSurface &surface = prepared[begin + relative];
    const PackedGrid &grid = host_grids[relative];
    SurfaceVoxelizationResult &result =
        *inputs[surface.input_index].result;
    result.records.clear();
    result.records.reserve(grid.cells / 8);
    for (int x = 0; x < grid.dimensions.x; ++x) {
      for (int y = 0; y < grid.dimensions.y; ++y) {
        for (int z = 0; z < grid.dimensions.z; ++z) {
          const size_t offset =
              grid.cell_offset +
              (static_cast<size_t>(x) * grid.dimensions.y + y) *
                  grid.dimensions.z +
              z;
          if (host_distances[offset] ==
              std::numeric_limits<unsigned long long>::max()) {
            continue;
          }
          double distance = 0.0;
          const unsigned long long bits = host_distances[offset];
          std::memcpy(&distance, &bits, sizeof(distance));
          result.records.push_back(
              {grid.minimum.x + x, grid.minimum.y + y,
               grid.minimum.z + z, distance, host_indices[offset]});
        }
      }
    }
    result.supported = true;
    result.fallback_reason.clear();
    result.elapsed_ms = milliseconds;
  }
  for (size_t index = begin; index < end; ++index) {
    const SurfaceVoxelizationInput &input =
        inputs[prepared[index].input_index];
    if (input.completion)
      input.completion();
  }
}

} // namespace

void voxelize_surfaces_batch(
    const std::vector<SurfaceVoxelizationInput> &inputs,
    ManifoldCudaBatchRuntime &runtime, size_t max_batch_size,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "manifold batch memory fraction must be in (0, 1]");
  }

  std::vector<PreparedSurface> prepared;
  prepared.reserve(inputs.size());
  for (size_t index = 0; index < inputs.size(); ++index) {
    const SurfaceVoxelizationInput &input = inputs[index];
    if (!input.mesh || !input.result) {
      throw std::invalid_argument(
          "manifold batch input contains a null pointer");
    }
    *input.result = SurfaceVoxelizationResult{};
    PreparedSurface surface;
    if (prepare_surface(input, index, surface)) {
      prepared.push_back(surface);
    } else if (input.completion) {
      input.completion();
    }
  }
  if (prepared.empty())
    return;

  const size_t wave_limit =
      max_batch_size == 0
          ? kMaximumAutomaticWave
          : std::min(max_batch_size, kMaximumAutomaticWave);
  size_t begin = 0;
  while (begin < prepared.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cuda_memory::check(cudaMemGetInfo(&free_bytes, &total_bytes),
                       "query batched manifold memory");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) *
                            memory_fraction);

    size_t end = begin;
    PackedCounts counts;
    while (end < prepared.size() && end - begin < wave_limit) {
      PackedCounts next;
      if (!append_counts(prepared[end], counts, next))
        break;
      if (packed_bytes(next) > budget)
        break;
      counts = next;
      ++end;
    }

    if (end == begin) {
      SurfaceVoxelizationResult &result =
          *inputs[prepared[begin].input_index].result;
      set_fallback(result,
                   "CUDA preprocessing memory budget exceeded");
      const SurfaceVoxelizationInput &input =
          inputs[prepared[begin].input_index];
      if (input.completion)
        input.completion();
      ++begin;
      continue;
    }
    run_wave(inputs, prepared, begin, end, counts, *runtime.impl_);
    begin = end;
  }
}

} // namespace neural_acd
