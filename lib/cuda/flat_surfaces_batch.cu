#include <algorithm>
#include <batch_executor.hpp>
#include <cmath>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <flat_surfaces_batch.hpp>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace neural_acd {
namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

size_t checked_multiply(size_t first, size_t second, const char *message) {
  if (first != 0 && second > std::numeric_limits<size_t>::max() / first)
    throw std::overflow_error(message);
  return first * second;
}

size_t checked_add(size_t first, size_t second, const char *message) {
  if (second > std::numeric_limits<size_t>::max() - first)
    throw std::overflow_error(message);
  return first + second;
}

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

size_t growth_bytes(const DeviceBuffer &buffer, size_t requested) {
  return requested > buffer.capacity() ? requested - buffer.capacity() : 0;
}

__device__ std::uint64_t edge_key(int first, int second) {
  const unsigned int low = static_cast<unsigned int>(min(first, second));
  const unsigned int high = static_cast<unsigned int>(max(first, second));
  return (static_cast<std::uint64_t>(low) << 32) | high;
}

__global__ void build_surface_features_kernel(
    const double3 *vertices, const int3 *triangles, double3 *normals,
    double *areas, std::uint64_t *edge_keys, int *edge_triangles,
    int triangle_count) {
  const int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangle_index >= triangle_count)
    return;

  const int3 triangle = triangles[triangle_index];
  const double3 first = vertices[triangle.x];
  const double3 second = vertices[triangle.y];
  const double3 third = vertices[triangle.z];
  const double first_x = second.x - first.x;
  const double first_y = second.y - first.y;
  const double first_z = second.z - first.z;
  const double second_x = third.x - first.x;
  const double second_y = third.y - first.y;
  const double second_z = third.z - first.z;
  const double normal_x = first_y * second_z - first_z * second_y;
  const double normal_y = first_z * second_x - first_x * second_z;
  const double normal_z = first_x * second_y - first_y * second_x;
  const double length =
      sqrt(normal_x * normal_x + normal_y * normal_y +
           normal_z * normal_z);
  normals[triangle_index] =
      make_double3(normal_x / length, normal_y / length,
                   normal_z / length);
  areas[triangle_index] = length * 0.5;

  const int edge_offset = triangle_index * 3;
  edge_keys[edge_offset] = edge_key(triangle.x, triangle.y);
  edge_keys[edge_offset + 1] = edge_key(triangle.y, triangle.z);
  edge_keys[edge_offset + 2] = edge_key(triangle.z, triangle.x);
  edge_triangles[edge_offset] = triangle_index;
  edge_triangles[edge_offset + 1] = triangle_index;
  edge_triangles[edge_offset + 2] = triangle_index;
}

} // namespace

struct FlatSurfaceBatchRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer triangles;
  DeviceBuffer normals;
  DeviceBuffer areas;
  DeviceBuffer edge_keys;
  DeviceBuffer sorted_edge_keys;
  DeviceBuffer edge_triangles;
  DeviceBuffer sorted_edge_triangles;
  DeviceBuffer sort_temp;
  PinnedBuffer host_vertices;
  PinnedBuffer host_triangles;
  PinnedBuffer host_normals;
  PinnedBuffer host_areas;
  PinnedBuffer host_input_edge_keys;
  PinnedBuffer host_edge_keys;
  PinnedBuffer host_edge_triangles;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags flat surfaces");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }

  size_t growth(size_t vertex_count, size_t triangle_count) const {
    const size_t edge_count = checked_multiply(
        triangle_count, 3, "Flat-surface edge count overflow");
    size_t result = 0;
    const auto include = [&](const DeviceBuffer &buffer, size_t count,
                             size_t element_size) {
      result = checked_add(
          result,
          growth_bytes(buffer,
                       checked_multiply(count, element_size,
                                        "Flat-surface allocation overflow")),
          "Flat-surface allocation total overflow");
    };
    include(vertices, vertex_count, sizeof(double3));
    include(triangles, triangle_count, sizeof(int3));
    include(normals, triangle_count, sizeof(double3));
    include(areas, triangle_count, sizeof(double));
    include(edge_keys, edge_count, sizeof(std::uint64_t));
    include(sorted_edge_keys, edge_count, sizeof(std::uint64_t));
    include(edge_triangles, edge_count, sizeof(int));
    include(sorted_edge_triangles, edge_count, sizeof(int));
    return result;
  }
};

FlatSurfaceBatchRuntime::FlatSurfaceBatchRuntime()
    : impl_(std::make_unique<Impl>()) {}
FlatSurfaceBatchRuntime::~FlatSurfaceBatchRuntime() = default;
FlatSurfaceBatchRuntime::FlatSurfaceBatchRuntime(
    FlatSurfaceBatchRuntime &&) noexcept = default;
FlatSurfaceBatchRuntime &
FlatSurfaceBatchRuntime::operator=(FlatSurfaceBatchRuntime &&) noexcept =
    default;

namespace {

void validate_inputs(const std::vector<FlatSurfaceBatchInput> &inputs) {
  for (const FlatSurfaceBatchInput &input : inputs) {
    if (!input.mesh || !input.surfaces) {
      throw std::invalid_argument(
          "Flat-surface batch input contains a null pointer");
    }
    if (input.mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.mesh->triangles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max() / 3)) {
      throw std::overflow_error(
          "Flat-surface mesh exceeds indexing limits");
    }
    for (const auto &triangle : input.mesh->triangles) {
      for (int vertex : triangle) {
        if (vertex < 0 ||
            static_cast<size_t>(vertex) >= input.mesh->vertices.size()) {
          throw std::invalid_argument(
              "Flat-surface mesh contains an invalid triangle index");
        }
      }
    }
  }
}

void run_wave(const std::vector<FlatSurfaceBatchInput> &inputs,
              size_t begin, size_t end,
              FlatSurfaceBatchRuntime::Impl &runtime,
              BatchExecutor *executor) {
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  for (size_t index = begin; index < end; ++index) {
    const Mesh &mesh = *inputs[index].mesh;
    if (mesh.vertices.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()) -
            vertex_count) {
      throw std::overflow_error(
          "Packed flat-surface vertices exceed indexing limits");
    }
    if (mesh.triangles.size() >
        static_cast<size_t>(std::numeric_limits<int>::max() / 3) -
            triangle_count) {
      throw std::overflow_error(
          "Packed flat-surface triangles exceed indexing limits");
    }
    vertex_count += mesh.vertices.size();
    triangle_count += mesh.triangles.size();
  }

  if (triangle_count == 0) {
    for (size_t index = begin; index < end; ++index)
      inputs[index].surfaces->clear();
    return;
  }

  const size_t edge_count = checked_multiply(
      triangle_count, 3, "Packed flat-surface edge count overflow");
  runtime.ensure_stream();
  runtime.vertices.ensure(vertex_count * sizeof(double3),
                          "cudaMalloc flat-surface vertices");
  runtime.triangles.ensure(triangle_count * sizeof(int3),
                           "cudaMalloc flat-surface triangles");
  runtime.normals.ensure(triangle_count * sizeof(double3),
                         "cudaMalloc flat-surface normals");
  runtime.areas.ensure(triangle_count * sizeof(double),
                       "cudaMalloc flat-surface areas");
  runtime.edge_keys.ensure(edge_count * sizeof(std::uint64_t),
                           "cudaMalloc flat-surface edge keys");
  runtime.sorted_edge_keys.ensure(
      edge_count * sizeof(std::uint64_t),
      "cudaMalloc sorted flat-surface edge keys");
  runtime.edge_triangles.ensure(edge_count * sizeof(int),
                                "cudaMalloc flat-surface edge triangles");
  runtime.sorted_edge_triangles.ensure(
      edge_count * sizeof(int),
      "cudaMalloc sorted flat-surface edge triangles");
  runtime.host_vertices.ensure(vertex_count * sizeof(double3),
                               "cudaMallocHost flat-surface vertices");
  runtime.host_triangles.ensure(triangle_count * sizeof(int3),
                                "cudaMallocHost flat-surface triangles");
  runtime.host_normals.ensure(triangle_count * sizeof(double3),
                              "cudaMallocHost flat-surface normals");
  runtime.host_areas.ensure(triangle_count * sizeof(double),
                            "cudaMallocHost flat-surface areas");
  runtime.host_input_edge_keys.ensure(
      edge_count * sizeof(std::uint64_t),
      "cudaMallocHost input flat-surface edge keys");
  runtime.host_edge_keys.ensure(
      edge_count * sizeof(std::uint64_t),
      "cudaMallocHost flat-surface edge keys");
  runtime.host_edge_triangles.ensure(
      edge_count * sizeof(int),
      "cudaMallocHost flat-surface edge triangles");

  double3 *host_vertices = runtime.host_vertices.as<double3>();
  int3 *host_triangles = runtime.host_triangles.as<int3>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  for (size_t index = begin; index < end; ++index) {
    const Mesh &mesh = *inputs[index].mesh;
    for (const Vec3D &vertex : mesh.vertices) {
      host_vertices[vertex_offset++] =
          make_double3(vertex[0], vertex[1], vertex[2]);
    }
    const size_t mesh_vertex_offset =
        vertex_offset - mesh.vertices.size();
    for (const auto &triangle : mesh.triangles) {
      host_triangles[triangle_offset++] = make_int3(
          static_cast<int>(mesh_vertex_offset) + triangle[0],
          static_cast<int>(mesh_vertex_offset) + triangle[1],
          static_cast<int>(mesh_vertex_offset) + triangle[2]);
    }
  }

  check_cuda(cudaMemcpyAsync(runtime.vertices.as<double3>(),
                             host_vertices,
                             vertex_count * sizeof(double3),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy flat-surface vertices");
  check_cuda(cudaMemcpyAsync(runtime.triangles.as<int3>(),
                             host_triangles,
                             triangle_count * sizeof(int3),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy flat-surface triangles");

  constexpr int block_size = 256;
  const int triangle_count_int = static_cast<int>(triangle_count);
  const int triangle_blocks =
      (triangle_count_int + block_size - 1) / block_size;
  build_surface_features_kernel<<<triangle_blocks, block_size, 0,
                                  runtime.stream>>>(
      runtime.vertices.as<double3>(), runtime.triangles.as<int3>(),
      runtime.normals.as<double3>(), runtime.areas.as<double>(),
      runtime.edge_keys.as<std::uint64_t>(),
      runtime.edge_triangles.as<int>(), triangle_count_int);
  check_cuda(cudaGetLastError(), "launch flat-surface features");

  size_t sort_temp_bytes = 0;
  const int edge_count_int = static_cast<int>(edge_count);
  check_cuda(cub::DeviceRadixSort::SortPairs(
                 nullptr, sort_temp_bytes,
                 runtime.edge_keys.as<std::uint64_t>(),
                 runtime.sorted_edge_keys.as<std::uint64_t>(),
                 runtime.edge_triangles.as<int>(),
                 runtime.sorted_edge_triangles.as<int>(), edge_count_int,
                 0, 64, runtime.stream),
             "query flat-surface radix sort storage");
  runtime.sort_temp.ensure(sort_temp_bytes,
                           "cudaMalloc flat-surface radix sort storage");
  check_cuda(cub::DeviceRadixSort::SortPairs(
                 runtime.sort_temp.as<void>(), sort_temp_bytes,
                 runtime.edge_keys.as<std::uint64_t>(),
                 runtime.sorted_edge_keys.as<std::uint64_t>(),
                 runtime.edge_triangles.as<int>(),
                 runtime.sorted_edge_triangles.as<int>(), edge_count_int,
                 0, 64, runtime.stream),
             "sort flat-surface edges");

  check_cuda(cudaMemcpyAsync(runtime.host_normals.as<double3>(),
                             runtime.normals.as<double3>(),
                             triangle_count * sizeof(double3),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy flat-surface normals");
  check_cuda(cudaMemcpyAsync(runtime.host_areas.as<double>(),
                             runtime.areas.as<double>(),
                             triangle_count * sizeof(double),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy flat-surface areas");
  check_cuda(cudaMemcpyAsync(
                 runtime.host_input_edge_keys.as<std::uint64_t>(),
                 runtime.edge_keys.as<std::uint64_t>(),
                 edge_count * sizeof(std::uint64_t),
                 cudaMemcpyDeviceToHost, runtime.stream),
             "copy input flat-surface edge keys");
  check_cuda(cudaMemcpyAsync(runtime.host_edge_keys.as<std::uint64_t>(),
                             runtime.sorted_edge_keys.as<std::uint64_t>(),
                             edge_count * sizeof(std::uint64_t),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy flat-surface edge keys");
  check_cuda(cudaMemcpyAsync(runtime.host_edge_triangles.as<int>(),
                             runtime.sorted_edge_triangles.as<int>(),
                             edge_count * sizeof(int),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy flat-surface edge triangles");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "cudaStreamSynchronize flat surfaces");

  const double3 *host_normals = runtime.host_normals.as<double3>();
  const double *host_areas = runtime.host_areas.as<double>();
  const std::uint64_t *host_input_edge_keys =
      runtime.host_input_edge_keys.as<std::uint64_t>();
  const std::uint64_t *host_edge_keys =
      runtime.host_edge_keys.as<std::uint64_t>();
  const int *host_edge_triangles =
      runtime.host_edge_triangles.as<int>();
  std::vector<size_t> triangle_offsets(end - begin + 1, 0);
  std::vector<size_t> edge_offsets(end - begin + 1, 0);
  for (size_t relative = 0; relative < end - begin; ++relative) {
    const size_t mesh_triangle_count =
        inputs[begin + relative].mesh->triangles.size();
    triangle_offsets[relative + 1] =
        triangle_offsets[relative] + mesh_triangle_count;
    edge_offsets[relative + 1] =
        edge_offsets[relative] + mesh_triangle_count * 3;
  }

  const auto assemble_input = [&](size_t relative) {
    const size_t index = begin + relative;
    const Mesh &mesh = *inputs[index].mesh;
    const size_t mesh_triangle_count = mesh.triangles.size();
    const size_t mesh_edge_count = mesh_triangle_count * 3;
    const size_t mesh_triangle_offset = triangle_offsets[relative];
    const size_t mesh_edge_offset = edge_offsets[relative];
    std::vector<Vec3D> normals(mesh_triangle_count);
    std::vector<double> areas(mesh_triangle_count);
    for (size_t triangle = 0; triangle < mesh_triangle_count; ++triangle) {
      const double3 normal =
          host_normals[mesh_triangle_offset + triangle];
      normals[triangle] = {normal.x, normal.y, normal.z};
      areas[triangle] =
          host_areas[mesh_triangle_offset + triangle];
    }
    std::vector<std::uint64_t> input_edge_keys(
        host_input_edge_keys + mesh_edge_offset,
        host_input_edge_keys + mesh_edge_offset + mesh_edge_count);
    std::vector<std::uint64_t> sorted_edge_keys(
        host_edge_keys + mesh_edge_offset,
        host_edge_keys + mesh_edge_offset + mesh_edge_count);
    std::vector<int> edge_triangles(mesh_edge_count);
    for (size_t edge = 0; edge < mesh_edge_count; ++edge) {
      edge_triangles[edge] =
          host_edge_triangles[mesh_edge_offset + edge] -
          static_cast<int>(mesh_triangle_offset);
    }
    *inputs[index].surfaces = assemble_surfaces_from_features(
        mesh, inputs[index].min_area, normals, areas, input_edge_keys,
        sorted_edge_keys, edge_triangles);
  };
  const size_t input_count = end - begin;
  if (executor && executor->thread_count() > 1 && input_count > 1) {
    executor->parallel_for_priority(input_count, assemble_input);
  } else {
    for (size_t relative = 0; relative < input_count; ++relative)
      assemble_input(relative);
  }
}

} // namespace

void extract_flat_surfaces_batch(
    const std::vector<FlatSurfaceBatchInput> &inputs,
    FlatSurfaceBatchRuntime &runtime, size_t max_batch_size,
    double memory_fraction, BatchExecutor *executor) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Flat-surface memory fraction must be in (0, 1]");
  }
  validate_inputs(inputs);

  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo flat surfaces");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) *
                            memory_fraction);
    size_t end = begin;
    size_t vertex_count = 0;
    size_t triangle_count = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const Mesh &mesh = *inputs[end].mesh;
      if (mesh.vertices.size() >
              static_cast<size_t>(std::numeric_limits<int>::max()) -
                  vertex_count ||
          mesh.triangles.size() >
              static_cast<size_t>(std::numeric_limits<int>::max() / 3) -
                  triangle_count) {
        break;
      }
      const size_t next_vertex_count =
          vertex_count + mesh.vertices.size();
      const size_t next_triangle_count =
          triangle_count + mesh.triangles.size();
      if (end > begin &&
          runtime.impl_->growth(next_vertex_count,
                                next_triangle_count) > budget) {
        break;
      }
      vertex_count = next_vertex_count;
      triangle_count = next_triangle_count;
      ++end;
    }
    if (end == begin)
      ++end;
    run_wave(inputs, begin, end, *runtime.impl_, executor);
    begin = end;
  }
}

} // namespace neural_acd
