#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <preprocess_cuda.hpp>

#include <algorithm>
#include <chrono>
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

struct DenseGrid {
  int3 minimum;
  int3 dimensions;
  size_t cells = 0;
};

struct SeedVoxel {
  int3 coordinate;
  int triangle_index;
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
  // OpenVDB intentionally evaluates (a, c, b).
  const double3 closest = closest_triangle(a, c, b, point);
  const double3 delta = subtract(point, closest);
  return dot_product(delta, delta);
}

__device__ size_t dense_offset(int x, int y, int z, int3 minimum,
                               int3 dimensions) {
  return (static_cast<size_t>(x - minimum.x) * dimensions.y +
          static_cast<size_t>(y - minimum.y)) *
             dimensions.z +
         static_cast<size_t>(z - minimum.z);
}

__global__ void find_seed_voxels_kernel(
    const float3 *vertices, const int3 *triangles, size_t triangle_count,
    SeedVoxel *seeds, unsigned int *seed_count, size_t seed_capacity) {
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
    if (output < seed_capacity)
      seeds[output] = {make_int3(x, y, z),
                       static_cast<int>(triangle_index)};
  }
}

template <bool StoreTriangles>
__device__ void store_voxel(double3 a, double3 b, double3 c, int x, int y,
                            int z, int triangle_index, int3 grid_minimum,
                            int3 grid_dimensions,
                            unsigned long long *distance_bits,
                            int *triangle_indices) {
  const double distance = squared_distance(a, b, c, x, y, z);
  const size_t offset = dense_offset(x, y, z, grid_minimum,
                                     grid_dimensions);
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
    const float3 *vertices, const int3 *triangles, const SeedVoxel *seeds,
    const unsigned int *seed_count, int3 grid_minimum, int3 grid_dimensions,
    unsigned long long *distance_bits, int *triangle_indices) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t seed_index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       seed_index < *seed_count; seed_index += stride) {
    const SeedVoxel seed = seeds[seed_index];
    const int3 triangle = triangles[seed.triangle_index];
    const float3 af = vertices[triangle.x];
    const float3 bf = vertices[triangle.y];
    const float3 cf = vertices[triangle.z];
    const double3 a = make_double3(af.x, af.y, af.z);
    const double3 b = make_double3(bf.x, bf.y, bf.z);
    const double3 c = make_double3(cf.x, cf.y, cf.z);
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          store_voxel<StoreTriangles>(
              a, b, c, seed.coordinate.x + dx, seed.coordinate.y + dy,
              seed.coordinate.z + dz, seed.triangle_index, grid_minimum,
              grid_dimensions, distance_bits, triangle_indices);
        }
      }
    }
  }
}

template <bool StoreTriangles>
__global__ void store_initial_neighbourhood_kernel(
    const float3 *vertices, const int3 *triangles, size_t triangle_count,
    int3 grid_minimum, int3 grid_dimensions,
    unsigned long long *distance_bits, int *triangle_indices) {
  const size_t triangle_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (triangle_index >= triangle_count)
    return;
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
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        store_voxel<StoreTriangles>(
            a, b, c, ax + dx, ay + dy, az + dz,
            static_cast<int>(triangle_index), grid_minimum,
            grid_dimensions, distance_bits, triangle_indices);
      }
    }
  }
}

bool finite_vertex(float3 vertex) {
  return std::isfinite(vertex.x) && std::isfinite(vertex.y) &&
         std::isfinite(vertex.z);
}

bool same_vertex(float3 first, float3 second) {
  return first.x == second.x && first.y == second.y &&
         first.z == second.z;
}

bool checked_multiply(size_t first, size_t second, size_t &result) {
  if (first != 0 && second > std::numeric_limits<size_t>::max() / first)
    return false;
  result = first * second;
  return true;
}

} // namespace

struct ManifoldCudaRuntime::Impl {
  cudaStream_t stream = nullptr;
  cudaEvent_t started = nullptr;
  cudaEvent_t finished = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer triangles;
  DeviceBuffer distance_bits;
  DeviceBuffer triangle_indices;
  DeviceBuffer seeds;
  DeviceBuffer seed_count;
  PinnedBuffer host_vertices;
  PinnedBuffer host_triangles;
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
          "create manifold preprocessing stream");
      cuda_memory::check(cudaEventCreate(&started),
                         "create manifold start event");
      cuda_memory::check(cudaEventCreate(&finished),
                         "create manifold finish event");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }
};

ManifoldCudaRuntime::ManifoldCudaRuntime()
    : impl_(std::make_unique<Impl>()) {}
ManifoldCudaRuntime::~ManifoldCudaRuntime() = default;
ManifoldCudaRuntime::ManifoldCudaRuntime(ManifoldCudaRuntime &&) noexcept =
    default;
ManifoldCudaRuntime &
ManifoldCudaRuntime::operator=(ManifoldCudaRuntime &&) noexcept = default;

SurfaceVoxelizationResult
ManifoldCudaRuntime::voxelize_surface(const Mesh &mesh, double scale,
                                      double memory_fraction) {
  SurfaceVoxelizationResult result;
  if (!std::isfinite(scale) || scale <= 0.0) {
    result.fallback_reason = "scale must be finite and positive";
    return result;
  }
  if (memory_fraction <= 0.0 || memory_fraction > 1.0)
    throw std::invalid_argument("memory_fraction must be in (0, 1]");
  if (mesh.vertices.empty() || mesh.triangles.empty()) {
    result.fallback_reason = "mesh is empty";
    return result;
  }
  if (mesh.vertices.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      mesh.triangles.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    result.fallback_reason = "mesh index range exceeds CUDA limits";
    return result;
  }

  Impl &runtime = *impl_;
  runtime.ensure_runtime();
  runtime.host_vertices.ensure(mesh.vertices.size() * sizeof(float3),
                               "allocate manifold input vertices");
  runtime.host_triangles.ensure(mesh.triangles.size() * sizeof(int3),
                                "allocate manifold input triangles");
  float3 *host_vertices = runtime.host_vertices.as<float3>();
  int3 *host_triangles = runtime.host_triangles.as<int3>();

  float3 minimum = make_float3(std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::infinity());
  float3 maximum = make_float3(-std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity());
  for (size_t index = 0; index < mesh.vertices.size(); ++index) {
    const Vec3D &source = mesh.vertices[index];
    const float3 vertex =
        make_float3(static_cast<float>(source[0] * scale),
                    static_cast<float>(source[1] * scale),
                    static_cast<float>(source[2] * scale));
    if (!finite_vertex(vertex)) {
      result.fallback_reason = "mesh contains non-finite vertices";
      return result;
    }
    host_vertices[index] = vertex;
    minimum.x = std::min(minimum.x, vertex.x);
    minimum.y = std::min(minimum.y, vertex.y);
    minimum.z = std::min(minimum.z, vertex.z);
    maximum.x = std::max(maximum.x, vertex.x);
    maximum.y = std::max(maximum.y, vertex.y);
    maximum.z = std::max(maximum.z, vertex.z);
  }

  size_t candidate_voxels = 0;
  for (size_t index = 0; index < mesh.triangles.size(); ++index) {
    const auto &source = mesh.triangles[index];
    if (source[0] < 0 || source[1] < 0 || source[2] < 0 ||
        static_cast<size_t>(source[0]) >= mesh.vertices.size() ||
        static_cast<size_t>(source[1]) >= mesh.vertices.size() ||
        static_cast<size_t>(source[2]) >= mesh.vertices.size()) {
      result.fallback_reason = "mesh contains invalid triangle indices";
      return result;
    }
    const int3 triangle = make_int3(source[0], source[1], source[2]);
    host_triangles[index] = triangle;
    const float3 a = host_vertices[triangle.x];
    const float3 b = host_vertices[triangle.y];
    const float3 c = host_vertices[triangle.z];
    if (same_vertex(a, b) || same_vertex(a, c) || same_vertex(b, c)) {
      result.fallback_reason = "mesh contains duplicate float vertices";
      return result;
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
    if (!checked_multiply(x, y, xy) || !checked_multiply(xy, z, xyz) ||
        xyz > std::numeric_limits<size_t>::max() - candidate_voxels) {
      result.fallback_reason = "voxel candidate count overflows";
      return result;
    }
    candidate_voxels += xyz;
  }
  result.candidate_voxels = candidate_voxels;
  if (candidate_voxels > std::numeric_limits<unsigned int>::max()) {
    result.fallback_reason = "seed voxel count exceeds CUDA limits";
    return result;
  }

  DenseGrid grid;
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
    result.fallback_reason = "dense voxel bounds overflow";
    return result;
  }

  size_t grid_bytes = 0;
  if (!checked_multiply(grid.cells,
                        sizeof(unsigned long long) + sizeof(int),
                        grid_bytes)) {
    result.fallback_reason = "dense voxel allocation overflows";
    return result;
  }
  const size_t input_bytes = mesh.vertices.size() * sizeof(float3) +
                             mesh.triangles.size() * sizeof(int3);
  size_t seed_bytes = 0;
  if (!checked_multiply(candidate_voxels, sizeof(SeedVoxel), seed_bytes) ||
      seed_bytes > std::numeric_limits<size_t>::max() -
                       sizeof(unsigned int)) {
    result.fallback_reason = "seed voxel allocation overflows";
    return result;
  }
  seed_bytes += sizeof(unsigned int);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cuda_memory::check(cudaMemGetInfo(&free_bytes, &total_bytes),
                     "query manifold preprocessing memory");
  const size_t budget = static_cast<size_t>(free_bytes * memory_fraction);
  if (grid_bytes > budget || input_bytes > budget - grid_bytes ||
      seed_bytes > budget - grid_bytes - input_bytes) {
    result.fallback_reason = "CUDA preprocessing memory budget exceeded";
    return result;
  }

  runtime.vertices.ensure(mesh.vertices.size() * sizeof(float3),
                          "allocate manifold device vertices");
  runtime.triangles.ensure(mesh.triangles.size() * sizeof(int3),
                           "allocate manifold device triangles");
  runtime.distance_bits.ensure(grid.cells * sizeof(unsigned long long),
                               "allocate manifold distance grid");
  runtime.triangle_indices.ensure(grid.cells * sizeof(int),
                                  "allocate manifold triangle grid");
  runtime.seeds.ensure(candidate_voxels * sizeof(SeedVoxel),
                       "allocate manifold seed voxels");
  runtime.seed_count.ensure(sizeof(unsigned int),
                            "allocate manifold seed count");
  runtime.host_distance_bits.ensure(grid.cells * sizeof(unsigned long long),
                                    "allocate manifold host distances");
  runtime.host_triangle_indices.ensure(grid.cells * sizeof(int),
                                       "allocate manifold host triangles");

  cuda_memory::check(cudaEventRecord(runtime.started, runtime.stream),
                     "record manifold preprocessing start");
  cuda_memory::check(
      cudaMemcpyAsync(runtime.vertices.as<float3>(), host_vertices,
                      mesh.vertices.size() * sizeof(float3),
                      cudaMemcpyHostToDevice, runtime.stream),
      "copy manifold vertices");
  cuda_memory::check(
      cudaMemcpyAsync(runtime.triangles.as<int3>(), host_triangles,
                      mesh.triangles.size() * sizeof(int3),
                      cudaMemcpyHostToDevice, runtime.stream),
      "copy manifold triangles");
  cuda_memory::check(
      cudaMemsetAsync(runtime.distance_bits.as<unsigned long long>(), 0xff,
                      grid.cells * sizeof(unsigned long long),
                      runtime.stream),
      "initialize manifold distance grid");
  cuda_memory::check(
      cudaMemsetAsync(runtime.triangle_indices.as<int>(), 0x7f,
                      grid.cells * sizeof(int), runtime.stream),
      "initialize manifold triangle grid");
  cuda_memory::check(
      cudaMemsetAsync(runtime.seed_count.as<unsigned int>(), 0,
                      sizeof(unsigned int), runtime.stream),
      "initialize manifold seed count");

  constexpr int threads = 128;
  find_seed_voxels_kernel<<<mesh.triangles.size(), threads, 0,
                             runtime.stream>>>(
      runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
      mesh.triangles.size(), runtime.seeds.as<SeedVoxel>(),
      runtime.seed_count.as<unsigned int>(), candidate_voxels);
  cuda_memory::check(cudaGetLastError(),
                     "launch manifold seed voxelization");
  const int seed_blocks = static_cast<int>(std::max<size_t>(
      1, std::min<size_t>(4096, (candidate_voxels + threads - 1) / threads)));
  const int triangle_blocks = static_cast<int>(
      (mesh.triangles.size() + threads - 1) / threads);
  store_initial_neighbourhood_kernel<false>
      <<<triangle_blocks, threads, 0, runtime.stream>>>(
          runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
          mesh.triangles.size(), grid.minimum, grid.dimensions,
          runtime.distance_bits.as<unsigned long long>(),
          runtime.triangle_indices.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "launch manifold initial distance halo");
  expand_seed_voxels_kernel<false>
      <<<seed_blocks, threads, 0, runtime.stream>>>(
          runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
          runtime.seeds.as<SeedVoxel>(),
          runtime.seed_count.as<unsigned int>(), grid.minimum,
          grid.dimensions,
          runtime.distance_bits.as<unsigned long long>(),
          runtime.triangle_indices.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "launch manifold distance voxelization");
  store_initial_neighbourhood_kernel<true>
      <<<triangle_blocks, threads, 0, runtime.stream>>>(
          runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
          mesh.triangles.size(), grid.minimum, grid.dimensions,
          runtime.distance_bits.as<unsigned long long>(),
          runtime.triangle_indices.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "launch manifold initial triangle halo");
  expand_seed_voxels_kernel<true>
      <<<seed_blocks, threads, 0, runtime.stream>>>(
          runtime.vertices.as<float3>(), runtime.triangles.as<int3>(),
          runtime.seeds.as<SeedVoxel>(),
          runtime.seed_count.as<unsigned int>(), grid.minimum,
          grid.dimensions,
          runtime.distance_bits.as<unsigned long long>(),
          runtime.triangle_indices.as<int>());
  cuda_memory::check(cudaGetLastError(),
                     "launch manifold triangle voxelization");

  cuda_memory::check(
      cudaMemcpyAsync(runtime.host_distance_bits.as<unsigned long long>(),
                      runtime.distance_bits.as<unsigned long long>(),
                      grid.cells * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost, runtime.stream),
      "copy manifold distance grid");
  cuda_memory::check(
      cudaMemcpyAsync(runtime.host_triangle_indices.as<int>(),
                      runtime.triangle_indices.as<int>(),
                      grid.cells * sizeof(int), cudaMemcpyDeviceToHost,
                      runtime.stream),
      "copy manifold triangle grid");
  cuda_memory::check(cudaEventRecord(runtime.finished, runtime.stream),
                     "record manifold preprocessing finish");
  cuda_memory::check(cudaEventSynchronize(runtime.finished),
                     "wait for manifold preprocessing");
  float milliseconds = 0.0f;
  cuda_memory::check(
      cudaEventElapsedTime(&milliseconds, runtime.started, runtime.finished),
      "time manifold preprocessing");
  result.elapsed_ms = milliseconds;

  const unsigned long long *host_distances =
      runtime.host_distance_bits.as<unsigned long long>();
  const int *host_indices = runtime.host_triangle_indices.as<int>();
  result.records.reserve(grid.cells / 8);
  for (int x = 0; x < grid.dimensions.x; ++x) {
    for (int y = 0; y < grid.dimensions.y; ++y) {
      for (int z = 0; z < grid.dimensions.z; ++z) {
        const size_t offset =
            (static_cast<size_t>(x) * grid.dimensions.y + y) *
                grid.dimensions.z +
            z;
        if (host_distances[offset] ==
            std::numeric_limits<unsigned long long>::max())
          continue;
        double distance = 0.0;
        const unsigned long long bits = host_distances[offset];
        std::memcpy(&distance, &bits, sizeof(distance));
        result.records.push_back(
            {grid.minimum.x + x, grid.minimum.y + y, grid.minimum.z + z,
             distance, host_indices[offset]});
      }
    }
  }
  result.supported = true;
  return result;
}

} // namespace neural_acd
