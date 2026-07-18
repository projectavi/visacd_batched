#include <algorithm>
#include <components_batch.hpp>
#include <cost.hpp>
#include <cub/cub.cuh>
#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <cmath>
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

__device__ unsigned long long edge_key(int first, int second) {
  const unsigned int low = static_cast<unsigned int>(min(first, second));
  const unsigned int high = static_cast<unsigned int>(max(first, second));
  return (static_cast<unsigned long long>(low) << 32) | high;
}

__global__ void build_edges_kernel(
    const int3 *triangles, unsigned long long *edge_keys,
    int *edge_triangles, int *parents, int triangle_count) {
  const int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangle_index >= triangle_count)
    return;
  const int3 triangle = triangles[triangle_index];
  parents[triangle_index] = triangle_index;
  const int edge_offset = triangle_index * 3;
  edge_keys[edge_offset] = edge_key(triangle.x, triangle.y);
  edge_keys[edge_offset + 1] = edge_key(triangle.y, triangle.z);
  edge_keys[edge_offset + 2] = edge_key(triangle.x, triangle.z);
  edge_triangles[edge_offset] = triangle_index;
  edge_triangles[edge_offset + 1] = triangle_index;
  edge_triangles[edge_offset + 2] = triangle_index;
}

__device__ int find_root(const int *parents, int value) {
  int parent = parents[value];
  while (parent != value) {
    value = parent;
    parent = parents[value];
  }
  return value;
}

__device__ void unite(int *parents, int first, int second) {
  while (true) {
    const int first_root = find_root(parents, first);
    const int second_root = find_root(parents, second);
    if (first_root == second_root)
      return;
    const int low = min(first_root, second_root);
    const int high = max(first_root, second_root);
    const int previous = atomicCAS(&parents[high], high, low);
    if (previous == high)
      return;
    first = low;
    second = previous;
  }
}

__global__ void union_edges_kernel(const unsigned long long *edge_keys,
                                   const int *edge_triangles, int *parents,
                                   int edge_count) {
  const int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index == 0 || edge_index >= edge_count)
    return;
  if (edge_keys[edge_index] == edge_keys[edge_index - 1]) {
    unite(parents, edge_triangles[edge_index],
          edge_triangles[edge_index - 1]);
  }
}

__global__ void compress_paths_kernel(int *parents, int triangle_count) {
  const int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangle_index >= triangle_count)
    return;
  parents[triangle_index] = find_root(parents, triangle_index);
}

__global__ void mark_component_roots_kernel(const int *parents,
                                            int *root_flags,
                                            int triangle_count) {
  const int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangle_index >= triangle_count)
    return;
  root_flags[triangle_index] = parents[triangle_index] == triangle_index;
}

__global__ void build_corner_keys_kernel(
    const int3 *triangles, const int *parents, const int *root_prefix,
    unsigned long long *corner_keys, int *corner_indices,
    int triangle_count) {
  const int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangle_index >= triangle_count)
    return;
  const int component = root_prefix[parents[triangle_index]];
  const int3 triangle = triangles[triangle_index];
  const int corner_offset = triangle_index * 3;
  corner_keys[corner_offset] =
      (static_cast<unsigned long long>(component) << 32) |
      static_cast<unsigned int>(triangle.x);
  corner_keys[corner_offset + 1] =
      (static_cast<unsigned long long>(component) << 32) |
      static_cast<unsigned int>(triangle.y);
  corner_keys[corner_offset + 2] =
      (static_cast<unsigned long long>(component) << 32) |
      static_cast<unsigned int>(triangle.z);
  corner_indices[corner_offset] = corner_offset;
  corner_indices[corner_offset + 1] = corner_offset + 1;
  corner_indices[corner_offset + 2] = corner_offset + 2;
}

__global__ void build_vertex_order_keys_kernel(
    const unsigned long long *unique_keys, const int *first_corners,
    unsigned long long *order_keys, int *unique_indices, int unique_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= unique_count)
    return;
  const unsigned int component =
      static_cast<unsigned int>(unique_keys[index] >> 32);
  order_keys[index] =
      (static_cast<unsigned long long>(component) << 32) |
      static_cast<unsigned int>(first_corners[index]);
  unique_indices[index] = index;
}

__device__ int component_lower_bound(const unsigned long long *keys,
                                     int count,
                                     unsigned int component) {
  int low = 0;
  int high = count;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    const unsigned int candidate =
        static_cast<unsigned int>(keys[middle] >> 32);
    if (candidate < component)
      low = middle + 1;
    else
      high = middle;
  }
  return low;
}

__device__ int key_lower_bound(const unsigned long long *keys, int count,
                               unsigned long long key) {
  int low = 0;
  int high = count;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (keys[middle] < key)
      low = middle + 1;
    else
      high = middle;
  }
  return low;
}

__global__ void assign_vertex_order_kernel(
    const unsigned long long *sorted_order_keys,
    const int *sorted_unique_indices, const unsigned long long *unique_keys,
    int *unique_local_ids, int *ordered_components,
    int *ordered_source_vertices, int unique_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= unique_count)
    return;
  const unsigned int component =
      static_cast<unsigned int>(sorted_order_keys[index] >> 32);
  const int component_begin = component_lower_bound(
      sorted_order_keys, unique_count, component);
  const int original_unique_index = sorted_unique_indices[index];
  unique_local_ids[original_unique_index] = index - component_begin;
  ordered_components[index] = static_cast<int>(component);
  ordered_source_vertices[index] =
      static_cast<int>(static_cast<unsigned int>(
          unique_keys[original_unique_index]));
}

__global__ void remap_triangles_kernel(
    const int3 *triangles, const int *parents, const int *root_prefix,
    const unsigned long long *unique_keys, const int *unique_local_ids,
    int *triangle_components, int3 *output_triangles, int unique_count,
    int triangle_count) {
  const int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangle_index >= triangle_count)
    return;
  const int component = root_prefix[parents[triangle_index]];
  const int3 triangle = triangles[triangle_index];
  const unsigned long long component_bits =
      static_cast<unsigned long long>(component) << 32;
  const unsigned long long first_key =
      component_bits | static_cast<unsigned int>(triangle.x);
  const unsigned long long second_key =
      component_bits | static_cast<unsigned int>(triangle.y);
  const unsigned long long third_key =
      component_bits | static_cast<unsigned int>(triangle.z);
  const int first = key_lower_bound(unique_keys, unique_count, first_key);
  const int second = key_lower_bound(unique_keys, unique_count, second_key);
  const int third = key_lower_bound(unique_keys, unique_count, third_key);
  triangle_components[triangle_index] = component;
  output_triangles[triangle_index] =
      make_int3(unique_local_ids[first], unique_local_ids[second],
                unique_local_ids[third]);
}

__global__ void remap_edges_kernel(
    const uint2 *edges, const int2 *triangle_ranges, const int *parents,
    const int *root_prefix, const unsigned long long *unique_keys,
    const int *unique_local_ids, int *edge_components, uint2 *output_edges,
    int unique_count, int edge_count) {
  const int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index >= edge_count)
    return;
  const uint2 edge = edges[edge_index];
  const int2 range = triangle_ranges[edge_index];
  int selected_component = -1;
  uint2 selected_edge = make_uint2(0, 0);
  for (int triangle = range.x; triangle < range.y; ++triangle) {
    if (parents[triangle] != triangle)
      continue;
    const int component = root_prefix[triangle];
    const unsigned long long component_bits =
        static_cast<unsigned long long>(component) << 32;
    const unsigned long long first_key = component_bits | edge.x;
    const unsigned long long second_key = component_bits | edge.y;
    const int first = key_lower_bound(unique_keys, unique_count, first_key);
    const int second = key_lower_bound(unique_keys, unique_count, second_key);
    if (first < unique_count && unique_keys[first] == first_key &&
        second < unique_count && unique_keys[second] == second_key) {
      selected_component = component;
      selected_edge = make_uint2(unique_local_ids[first],
                                 unique_local_ids[second]);
      break;
    }
  }
  edge_components[edge_index] = selected_component;
  output_edges[edge_index] = selected_edge;
}

size_t growth_bytes(const DeviceBuffer &buffer, size_t requested) {
  return requested > buffer.capacity() ? requested - buffer.capacity() : 0;
}

} // namespace

struct ComponentBatchRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer triangles;
  DeviceBuffer edge_keys;
  DeviceBuffer sorted_edge_keys;
  DeviceBuffer edge_triangles;
  DeviceBuffer sorted_edge_triangles;
  DeviceBuffer parents;
  DeviceBuffer root_flags;
  DeviceBuffer root_prefix;
  DeviceBuffer unique_keys;
  DeviceBuffer unique_first_corners;
  DeviceBuffer unique_count;
  DeviceBuffer unique_local_ids;
  DeviceBuffer ordered_components;
  DeviceBuffer ordered_source_vertices;
  DeviceBuffer triangle_components;
  DeviceBuffer output_triangles;
  DeviceBuffer edges;
  DeviceBuffer edge_triangle_ranges;
  DeviceBuffer edge_components;
  DeviceBuffer output_edges;
  DeviceBuffer sort_temp;
  PinnedBuffer host_triangles;
  PinnedBuffer host_labels;
  PinnedBuffer host_counts;
  PinnedBuffer host_ordered_components;
  PinnedBuffer host_ordered_source_vertices;
  PinnedBuffer host_triangle_components;
  PinnedBuffer host_output_triangles;
  PinnedBuffer host_edges;
  PinnedBuffer host_edge_triangle_ranges;
  PinnedBuffer host_edge_components;
  PinnedBuffer host_output_edges;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags components");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }

  size_t growth(size_t triangle_count) const {
    const size_t edge_count = checked_multiply(
        triangle_count, 3, "Component edge count overflow");
    size_t result = 0;
    const auto include = [&](const DeviceBuffer &buffer, size_t count,
                             size_t element_size) {
      result = checked_add(
          result,
          growth_bytes(buffer,
                       checked_multiply(count, element_size,
                                        "Component allocation overflow")),
          "Component allocation total overflow");
    };
    include(triangles, triangle_count, sizeof(int3));
    include(edge_keys, edge_count, sizeof(unsigned long long));
    include(sorted_edge_keys, edge_count, sizeof(unsigned long long));
    include(edge_triangles, edge_count, sizeof(int));
    include(sorted_edge_triangles, edge_count, sizeof(int));
    include(parents, triangle_count, sizeof(int));
    include(root_flags, triangle_count, sizeof(int));
    include(root_prefix, triangle_count, sizeof(int));
    include(unique_keys, edge_count, sizeof(unsigned long long));
    include(unique_first_corners, edge_count, sizeof(int));
    include(unique_local_ids, edge_count, sizeof(int));
    include(ordered_components, edge_count, sizeof(int));
    include(ordered_source_vertices, edge_count, sizeof(int));
    include(triangle_components, triangle_count, sizeof(int));
    include(output_triangles, triangle_count, sizeof(int3));
    return result;
  }
};

ComponentBatchRuntime::ComponentBatchRuntime()
    : impl_(std::make_unique<Impl>()) {}
ComponentBatchRuntime::~ComponentBatchRuntime() = default;
ComponentBatchRuntime::ComponentBatchRuntime(
    ComponentBatchRuntime &&) noexcept = default;
ComponentBatchRuntime &
ComponentBatchRuntime::operator=(ComponentBatchRuntime &&) noexcept = default;

namespace {

void validate_inputs(const std::vector<ComponentBatchInput> &inputs,
                     bool assemble) {
  for (const ComponentBatchInput &input : inputs) {
    if (!input.mesh || (assemble ? !input.components : !input.labels))
      throw std::invalid_argument("Component input contains a null pointer");
    if (input.mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.mesh->triangles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max() / 3)) {
      throw std::overflow_error("Component mesh exceeds indexing limits");
    }
    for (const auto &triangle : input.mesh->triangles) {
      for (int vertex : triangle) {
        if (vertex < 0 ||
            static_cast<size_t>(vertex) >= input.mesh->vertices.size()) {
          throw std::invalid_argument(
              "Component mesh contains an invalid triangle index");
        }
      }
    }
  }
}

void run_wave(const std::vector<ComponentBatchInput> &inputs, size_t begin,
              size_t end, ComponentBatchRuntime::Impl &runtime,
              bool assemble) {
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  size_t intersecting_edge_count = 0;
  for (size_t index = begin; index < end; ++index) {
    const Mesh &mesh = *inputs[index].mesh;
    if (mesh.vertices.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()) -
            vertex_count) {
      throw std::overflow_error("Packed component vertices exceed limits");
    }
    if (mesh.triangles.size() >
        static_cast<size_t>(std::numeric_limits<int>::max() / 3) -
            triangle_count) {
      throw std::overflow_error("Packed component triangles exceed limits");
    }
    vertex_count += mesh.vertices.size();
    triangle_count += mesh.triangles.size();
    if (assemble && !mesh.triangles.empty()) {
      if (mesh.intersecting_edges.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) -
              intersecting_edge_count) {
        throw std::overflow_error("Packed component edges exceed limits");
      }
      intersecting_edge_count += mesh.intersecting_edges.size();
    }
  }
  if (triangle_count == 0) {
    for (size_t index = begin; index < end; ++index) {
      if (assemble)
        inputs[index].components->clear();
      else
        inputs[index].labels->clear();
    }
    return;
  }

  const size_t edge_count = checked_multiply(
      triangle_count, 3, "Packed component edges overflow");
  const int edge_count_int = static_cast<int>(edge_count);
  runtime.ensure_stream();
  runtime.triangles.ensure(triangle_count * sizeof(int3),
                           "cudaMalloc component triangles");
  runtime.edge_keys.ensure(edge_count * sizeof(unsigned long long),
                           "cudaMalloc component edge keys");
  runtime.sorted_edge_keys.ensure(edge_count * sizeof(unsigned long long),
                                  "cudaMalloc sorted component edge keys");
  runtime.edge_triangles.ensure(edge_count * sizeof(int),
                                "cudaMalloc component edge triangles");
  runtime.sorted_edge_triangles.ensure(
      edge_count * sizeof(int),
      "cudaMalloc sorted component edge triangles");
  runtime.parents.ensure(triangle_count * sizeof(int),
                         "cudaMalloc component parents");
  runtime.host_triangles.ensure(triangle_count * sizeof(int3),
                                "cudaMallocHost component triangles");
  runtime.host_labels.ensure(triangle_count * sizeof(int),
                             "cudaMallocHost component labels");
  if (assemble) {
    runtime.root_flags.ensure(triangle_count * sizeof(int),
                              "cudaMalloc component root flags");
    runtime.root_prefix.ensure(triangle_count * sizeof(int),
                               "cudaMalloc component root prefix");
    runtime.unique_keys.ensure(edge_count * sizeof(unsigned long long),
                               "cudaMalloc component unique keys");
    runtime.unique_first_corners.ensure(
        edge_count * sizeof(int),
        "cudaMalloc component unique first corners");
    runtime.unique_count.ensure(sizeof(int),
                                "cudaMalloc component unique count");
    runtime.unique_local_ids.ensure(
        edge_count * sizeof(int),
        "cudaMalloc component unique local ids");
    runtime.ordered_components.ensure(
        edge_count * sizeof(int),
        "cudaMalloc ordered component ids");
    runtime.ordered_source_vertices.ensure(
        edge_count * sizeof(int),
        "cudaMalloc ordered component vertices");
    runtime.triangle_components.ensure(
        triangle_count * sizeof(int),
        "cudaMalloc triangle component ids");
    runtime.output_triangles.ensure(
        triangle_count * sizeof(int3),
        "cudaMalloc compacted component triangles");
    runtime.host_counts.ensure(3 * sizeof(int),
                               "cudaMallocHost component counts");
    runtime.host_ordered_components.ensure(
        edge_count * sizeof(int),
        "cudaMallocHost ordered component ids");
    runtime.host_ordered_source_vertices.ensure(
        edge_count * sizeof(int),
        "cudaMallocHost ordered component vertices");
    runtime.host_triangle_components.ensure(
        triangle_count * sizeof(int),
        "cudaMallocHost triangle component ids");
    runtime.host_output_triangles.ensure(
        triangle_count * sizeof(int3),
        "cudaMallocHost compacted component triangles");
    if (intersecting_edge_count > 0) {
      runtime.edges.ensure(intersecting_edge_count * sizeof(uint2),
                           "cudaMalloc component intersecting edges");
      runtime.edge_triangle_ranges.ensure(
          intersecting_edge_count * sizeof(int2),
          "cudaMalloc component edge triangle ranges");
      runtime.edge_components.ensure(
          intersecting_edge_count * sizeof(int),
          "cudaMalloc component edge assignments");
      runtime.output_edges.ensure(
          intersecting_edge_count * sizeof(uint2),
          "cudaMalloc compacted component edges");
      runtime.host_edges.ensure(
          intersecting_edge_count * sizeof(uint2),
          "cudaMallocHost component intersecting edges");
      runtime.host_edge_triangle_ranges.ensure(
          intersecting_edge_count * sizeof(int2),
          "cudaMallocHost component edge triangle ranges");
      runtime.host_edge_components.ensure(
          intersecting_edge_count * sizeof(int),
          "cudaMallocHost component edge assignments");
      runtime.host_output_edges.ensure(
          intersecting_edge_count * sizeof(uint2),
          "cudaMallocHost compacted component edges");
    }
  }

  int3 *host_triangles = runtime.host_triangles.as<int3>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  size_t intersecting_edge_offset = 0;
  uint2 *host_edges = assemble ? runtime.host_edges.as<uint2>() : nullptr;
  int2 *host_edge_triangle_ranges =
      assemble ? runtime.host_edge_triangle_ranges.as<int2>() : nullptr;
  for (size_t index = begin; index < end; ++index) {
    const Mesh &mesh = *inputs[index].mesh;
    const size_t mesh_triangle_begin = triangle_offset;
    for (const auto &triangle : mesh.triangles) {
      host_triangles[triangle_offset++] = make_int3(
          static_cast<int>(vertex_offset) + triangle[0],
          static_cast<int>(vertex_offset) + triangle[1],
          static_cast<int>(vertex_offset) + triangle[2]);
    }
    if (assemble && !mesh.triangles.empty()) {
      const int2 triangle_range = make_int2(
          static_cast<int>(mesh_triangle_begin),
          static_cast<int>(triangle_offset));
      for (const auto &edge : mesh.intersecting_edges) {
        host_edges[intersecting_edge_offset] = make_uint2(
            static_cast<unsigned int>(vertex_offset) + edge.first,
            static_cast<unsigned int>(vertex_offset) + edge.second);
        host_edge_triangle_ranges[intersecting_edge_offset] = triangle_range;
        ++intersecting_edge_offset;
      }
    }
    vertex_offset += mesh.vertices.size();
  }

  check_cuda(cudaMemcpyAsync(runtime.triangles.as<int3>(), host_triangles,
                             triangle_count * sizeof(int3),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy component triangles");
  constexpr int block_size = 256;
  const int triangle_blocks =
      (static_cast<int>(triangle_count) + block_size - 1) / block_size;
  build_edges_kernel<<<triangle_blocks, block_size, 0, runtime.stream>>>(
      runtime.triangles.as<int3>(),
      runtime.edge_keys.as<unsigned long long>(),
      runtime.edge_triangles.as<int>(), runtime.parents.as<int>(),
      static_cast<int>(triangle_count));
  check_cuda(cudaGetLastError(), "launch component edge construction");

  size_t sort_temp_bytes = 0;
  check_cuda(cub::DeviceRadixSort::SortPairs(
                 nullptr, sort_temp_bytes,
                 runtime.edge_keys.as<unsigned long long>(),
                 runtime.sorted_edge_keys.as<unsigned long long>(),
                 runtime.edge_triangles.as<int>(),
                 runtime.sorted_edge_triangles.as<int>(), edge_count_int, 0,
                 64,
                 runtime.stream),
             "query component radix sort storage");
  runtime.sort_temp.ensure(sort_temp_bytes,
                           "cudaMalloc component radix sort storage");
  check_cuda(cub::DeviceRadixSort::SortPairs(
                 runtime.sort_temp.as<void>(), sort_temp_bytes,
                 runtime.edge_keys.as<unsigned long long>(),
                 runtime.sorted_edge_keys.as<unsigned long long>(),
                 runtime.edge_triangles.as<int>(),
                 runtime.sorted_edge_triangles.as<int>(), edge_count_int, 0,
                 64,
                 runtime.stream),
             "sort component edges");

  const int edge_blocks =
      (edge_count_int + block_size - 1) / block_size;
  union_edges_kernel<<<edge_blocks, block_size, 0, runtime.stream>>>(
      runtime.sorted_edge_keys.as<unsigned long long>(),
      runtime.sorted_edge_triangles.as<int>(), runtime.parents.as<int>(),
      edge_count_int);
  check_cuda(cudaGetLastError(), "launch component unions");
  compress_paths_kernel<<<triangle_blocks, block_size, 0, runtime.stream>>>(
      runtime.parents.as<int>(), static_cast<int>(triangle_count));
  check_cuda(cudaGetLastError(), "launch component path compression");

  if (assemble) {
    mark_component_roots_kernel<<<triangle_blocks, block_size, 0,
                                  runtime.stream>>>(
        runtime.parents.as<int>(), runtime.root_flags.as<int>(),
        static_cast<int>(triangle_count));
    check_cuda(cudaGetLastError(), "launch component root marking");

    size_t scan_temp_bytes = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(
                   nullptr, scan_temp_bytes, runtime.root_flags.as<int>(),
                   runtime.root_prefix.as<int>(),
                   static_cast<int>(triangle_count), runtime.stream),
               "query component prefix scan storage");
    runtime.sort_temp.ensure(scan_temp_bytes,
                             "cudaMalloc component prefix scan storage");
    check_cuda(cub::DeviceScan::ExclusiveSum(
                   runtime.sort_temp.as<void>(), scan_temp_bytes,
                   runtime.root_flags.as<int>(),
                   runtime.root_prefix.as<int>(),
                   static_cast<int>(triangle_count), runtime.stream),
               "scan component roots");

    build_corner_keys_kernel<<<triangle_blocks, block_size, 0,
                               runtime.stream>>>(
        runtime.triangles.as<int3>(), runtime.parents.as<int>(),
        runtime.root_prefix.as<int>(),
        runtime.edge_keys.as<unsigned long long>(),
        runtime.edge_triangles.as<int>(),
        static_cast<int>(triangle_count));
    check_cuda(cudaGetLastError(), "launch component corner keys");

    size_t corner_sort_temp_bytes = 0;
    check_cuda(cub::DeviceRadixSort::SortPairs(
                   nullptr, corner_sort_temp_bytes,
                   runtime.edge_keys.as<unsigned long long>(),
                   runtime.sorted_edge_keys.as<unsigned long long>(),
                   runtime.edge_triangles.as<int>(),
                   runtime.sorted_edge_triangles.as<int>(), edge_count_int,
                   0, 64, runtime.stream),
               "query component corner sort storage");
    runtime.sort_temp.ensure(corner_sort_temp_bytes,
                             "cudaMalloc component corner sort storage");
    check_cuda(cub::DeviceRadixSort::SortPairs(
                   runtime.sort_temp.as<void>(), corner_sort_temp_bytes,
                   runtime.edge_keys.as<unsigned long long>(),
                   runtime.sorted_edge_keys.as<unsigned long long>(),
                   runtime.edge_triangles.as<int>(),
                   runtime.sorted_edge_triangles.as<int>(), edge_count_int,
                   0, 64, runtime.stream),
               "sort component corners");

    size_t reduce_temp_bytes = 0;
    check_cuda(cub::DeviceReduce::ReduceByKey(
                   nullptr, reduce_temp_bytes,
                   runtime.sorted_edge_keys.as<unsigned long long>(),
                   runtime.unique_keys.as<unsigned long long>(),
                   runtime.sorted_edge_triangles.as<int>(),
                   runtime.unique_first_corners.as<int>(),
                   runtime.unique_count.as<int>(), cub::Min(), edge_count_int,
                   runtime.stream),
               "query component vertex reduction storage");
    runtime.sort_temp.ensure(
        reduce_temp_bytes,
        "cudaMalloc component vertex reduction storage");
    check_cuda(cub::DeviceReduce::ReduceByKey(
                   runtime.sort_temp.as<void>(), reduce_temp_bytes,
                   runtime.sorted_edge_keys.as<unsigned long long>(),
                   runtime.unique_keys.as<unsigned long long>(),
                   runtime.sorted_edge_triangles.as<int>(),
                   runtime.unique_first_corners.as<int>(),
                   runtime.unique_count.as<int>(), cub::Min(), edge_count_int,
                   runtime.stream),
               "reduce component vertices");

    int *host_counts = runtime.host_counts.as<int>();
    check_cuda(cudaMemcpyAsync(host_counts, runtime.unique_count.as<int>(),
                               sizeof(int), cudaMemcpyDeviceToHost,
                               runtime.stream),
               "copy unique component vertex count");
    check_cuda(cudaMemcpyAsync(
                   host_counts + 1,
                   runtime.root_prefix.as<int>() + triangle_count - 1,
                   sizeof(int), cudaMemcpyDeviceToHost, runtime.stream),
               "copy component prefix tail");
    check_cuda(cudaMemcpyAsync(
                   host_counts + 2,
                   runtime.root_flags.as<int>() + triangle_count - 1,
                   sizeof(int), cudaMemcpyDeviceToHost, runtime.stream),
               "copy component root tail");
    check_cuda(cudaStreamSynchronize(runtime.stream),
               "synchronize component compact counts");

    const int unique_vertex_count = host_counts[0];
    const int component_count = host_counts[1] + host_counts[2];
    if (unique_vertex_count <= 0 || component_count <= 0)
      throw std::runtime_error("GPU component compaction produced no data");
    const int unique_blocks =
        (unique_vertex_count + block_size - 1) / block_size;
    build_vertex_order_keys_kernel<<<unique_blocks, block_size, 0,
                                     runtime.stream>>>(
        runtime.unique_keys.as<unsigned long long>(),
        runtime.unique_first_corners.as<int>(),
        runtime.edge_keys.as<unsigned long long>(),
        runtime.edge_triangles.as<int>(), unique_vertex_count);
    check_cuda(cudaGetLastError(), "launch component vertex order keys");

    size_t order_sort_temp_bytes = 0;
    check_cuda(cub::DeviceRadixSort::SortPairs(
                   nullptr, order_sort_temp_bytes,
                   runtime.edge_keys.as<unsigned long long>(),
                   runtime.sorted_edge_keys.as<unsigned long long>(),
                   runtime.edge_triangles.as<int>(),
                   runtime.sorted_edge_triangles.as<int>(),
                   unique_vertex_count, 0, 64, runtime.stream),
               "query component vertex order sort storage");
    runtime.sort_temp.ensure(
        order_sort_temp_bytes,
        "cudaMalloc component vertex order sort storage");
    check_cuda(cub::DeviceRadixSort::SortPairs(
                   runtime.sort_temp.as<void>(), order_sort_temp_bytes,
                   runtime.edge_keys.as<unsigned long long>(),
                   runtime.sorted_edge_keys.as<unsigned long long>(),
                   runtime.edge_triangles.as<int>(),
                   runtime.sorted_edge_triangles.as<int>(),
                   unique_vertex_count, 0, 64, runtime.stream),
               "sort component vertices by first occurrence");

    assign_vertex_order_kernel<<<unique_blocks, block_size, 0,
                                 runtime.stream>>>(
        runtime.sorted_edge_keys.as<unsigned long long>(),
        runtime.sorted_edge_triangles.as<int>(),
        runtime.unique_keys.as<unsigned long long>(),
        runtime.unique_local_ids.as<int>(),
        runtime.ordered_components.as<int>(),
        runtime.ordered_source_vertices.as<int>(), unique_vertex_count);
    check_cuda(cudaGetLastError(), "launch component vertex remap");
    remap_triangles_kernel<<<triangle_blocks, block_size, 0,
                             runtime.stream>>>(
        runtime.triangles.as<int3>(), runtime.parents.as<int>(),
        runtime.root_prefix.as<int>(),
        runtime.unique_keys.as<unsigned long long>(),
        runtime.unique_local_ids.as<int>(),
        runtime.triangle_components.as<int>(),
        runtime.output_triangles.as<int3>(), unique_vertex_count,
        static_cast<int>(triangle_count));
    check_cuda(cudaGetLastError(), "launch component triangle remap");

    if (intersecting_edge_count > 0) {
      check_cuda(cudaMemcpyAsync(
                     runtime.edges.as<uint2>(), host_edges,
                     intersecting_edge_count * sizeof(uint2),
                     cudaMemcpyHostToDevice, runtime.stream),
                 "copy component intersecting edges");
      check_cuda(cudaMemcpyAsync(
                     runtime.edge_triangle_ranges.as<int2>(),
                     host_edge_triangle_ranges,
                     intersecting_edge_count * sizeof(int2),
                     cudaMemcpyHostToDevice, runtime.stream),
                 "copy component edge triangle ranges");
      const int intersecting_edge_blocks =
          (static_cast<int>(intersecting_edge_count) + block_size - 1) /
          block_size;
      remap_edges_kernel<<<intersecting_edge_blocks, block_size, 0,
                           runtime.stream>>>(
          runtime.edges.as<uint2>(), runtime.edge_triangle_ranges.as<int2>(),
          runtime.parents.as<int>(), runtime.root_prefix.as<int>(),
          runtime.unique_keys.as<unsigned long long>(),
          runtime.unique_local_ids.as<int>(),
          runtime.edge_components.as<int>(), runtime.output_edges.as<uint2>(),
          unique_vertex_count, static_cast<int>(intersecting_edge_count));
      check_cuda(cudaGetLastError(), "launch component edge remap");
    }

    check_cuda(cudaMemcpyAsync(
                   runtime.host_ordered_components.as<int>(),
                   runtime.ordered_components.as<int>(),
                   unique_vertex_count * sizeof(int), cudaMemcpyDeviceToHost,
                   runtime.stream),
               "copy ordered component ids");
    check_cuda(cudaMemcpyAsync(
                   runtime.host_ordered_source_vertices.as<int>(),
                   runtime.ordered_source_vertices.as<int>(),
                   unique_vertex_count * sizeof(int), cudaMemcpyDeviceToHost,
                   runtime.stream),
               "copy ordered component vertices");
    check_cuda(cudaMemcpyAsync(
                   runtime.host_triangle_components.as<int>(),
                   runtime.triangle_components.as<int>(),
                   triangle_count * sizeof(int), cudaMemcpyDeviceToHost,
                   runtime.stream),
               "copy triangle component ids");
    check_cuda(cudaMemcpyAsync(
                   runtime.host_output_triangles.as<int3>(),
                   runtime.output_triangles.as<int3>(),
                   triangle_count * sizeof(int3), cudaMemcpyDeviceToHost,
                   runtime.stream),
               "copy compacted component triangles");
    if (intersecting_edge_count > 0) {
      check_cuda(cudaMemcpyAsync(
                     runtime.host_edge_components.as<int>(),
                     runtime.edge_components.as<int>(),
                     intersecting_edge_count * sizeof(int),
                     cudaMemcpyDeviceToHost, runtime.stream),
                 "copy component edge assignments");
      check_cuda(cudaMemcpyAsync(
                     runtime.host_output_edges.as<uint2>(),
                     runtime.output_edges.as<uint2>(),
                     intersecting_edge_count * sizeof(uint2),
                     cudaMemcpyDeviceToHost, runtime.stream),
                 "copy compacted component edges");
    }
    check_cuda(cudaStreamSynchronize(runtime.stream),
               "cudaStreamSynchronize component compaction");

    const int *ordered_components =
        runtime.host_ordered_components.as<int>();
    const int *ordered_source_vertices =
        runtime.host_ordered_source_vertices.as<int>();
    const int *triangle_components =
        runtime.host_triangle_components.as<int>();
    const int3 *output_triangles =
        runtime.host_output_triangles.as<int3>();
    const int *edge_components = runtime.host_edge_components.as<int>();
    const uint2 *output_edges = runtime.host_output_edges.as<uint2>();

    const size_t wave_size = end - begin;
    std::vector<size_t> vertex_offsets(wave_size + 1, 0);
    std::vector<size_t> triangle_offsets(wave_size + 1, 0);
    for (size_t local = 0; local < wave_size; ++local) {
      const Mesh &mesh = *inputs[begin + local].mesh;
      vertex_offsets[local + 1] =
          vertex_offsets[local] + mesh.vertices.size();
      triangle_offsets[local + 1] =
          triangle_offsets[local] + mesh.triangles.size();
    }

    std::vector<MeshList> assembled(wave_size);
    std::vector<size_t> component_inputs(component_count, wave_size);
    std::vector<size_t> component_locals(component_count, 0);
    for (size_t local = 0; local < wave_size; ++local) {
      const size_t triangle_begin = triangle_offsets[local];
      const size_t triangle_end = triangle_offsets[local + 1];
      if (triangle_begin == triangle_end)
        continue;
      int first_component = triangle_components[triangle_begin];
      int last_component = first_component;
      for (size_t triangle = triangle_begin; triangle < triangle_end;
           ++triangle) {
        last_component = std::max(last_component,
                                  triangle_components[triangle]);
      }
      const size_t count =
          static_cast<size_t>(last_component - first_component + 1);
      assembled[local].resize(count);
      for (size_t component = 0; component < count; ++component) {
        const size_t global_component =
            static_cast<size_t>(first_component) + component;
        component_inputs[global_component] = local;
        component_locals[global_component] = component;
      }
    }

    for (int vertex = 0; vertex < unique_vertex_count; ++vertex) {
      const size_t component =
          static_cast<size_t>(ordered_components[vertex]);
      const size_t local = component_inputs[component];
      if (local >= wave_size)
        throw std::runtime_error("Invalid compacted component owner");
      Mesh &output = assembled[local][component_locals[component]];
      const size_t source_vertex =
          static_cast<size_t>(ordered_source_vertices[vertex]) -
          vertex_offsets[local];
      const Mesh &source = *inputs[begin + local].mesh;
      output.vertices.push_back(source.vertices[source_vertex]);
      if (!source.is_new.empty())
        output.is_new.push_back(source.is_new[source_vertex]);
    }

    for (size_t local = 0; local < wave_size; ++local) {
      const size_t triangle_begin = triangle_offsets[local];
      const size_t triangle_end = triangle_offsets[local + 1];
      for (size_t triangle = triangle_begin; triangle < triangle_end;
           ++triangle) {
        const size_t component =
            static_cast<size_t>(triangle_components[triangle]);
        const int3 compacted = output_triangles[triangle];
        assembled[local][component_locals[component]].triangles.push_back(
            {compacted.x, compacted.y, compacted.z});
      }
    }

    size_t edge = 0;
    for (size_t local = 0; local < wave_size; ++local) {
      const Mesh &source = *inputs[begin + local].mesh;
      if (source.triangles.empty())
        continue;
      for (size_t source_edge = 0;
           source_edge < source.intersecting_edges.size(); ++source_edge) {
        const int component = edge_components[edge];
        if (component >= 0) {
          const uint2 compacted = output_edges[edge];
          assembled[local][component_locals[component]]
              .intersecting_edges.push_back({compacted.x, compacted.y});
        }
        ++edge;
      }
    }

    for (size_t local = 0; local < wave_size; ++local) {
      MeshList filtered;
      filtered.reserve(assembled[local].size());
      for (Mesh &part : assembled[local]) {
        if (std::abs(get_mesh_volume(part)) < 1e-6)
          continue;
        filtered.push_back(std::move(part));
      }
      *inputs[begin + local].components = std::move(filtered);
    }
    return;
  }

  check_cuda(cudaMemcpyAsync(runtime.host_labels.as<int>(),
                             runtime.parents.as<int>(),
                             triangle_count * sizeof(int),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy component labels");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "cudaStreamSynchronize components");

  const int *host_labels = runtime.host_labels.as<int>();
  triangle_offset = 0;
  for (size_t index = begin; index < end; ++index) {
    const size_t count = inputs[index].mesh->triangles.size();
    inputs[index].labels->assign(host_labels + triangle_offset,
                                 host_labels + triangle_offset + count);
    triangle_offset += count;
  }
}

} // namespace

void label_components_batch(const std::vector<ComponentBatchInput> &inputs,
                            ComponentBatchRuntime &runtime,
                            size_t max_batch_size,
                            double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Component memory fraction must be in (0, 1]");
  }
  validate_inputs(inputs, false);

  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo components");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
    size_t end = begin;
    size_t triangle_count = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const size_t addition = inputs[end].mesh->triangles.size();
      if (addition > std::numeric_limits<size_t>::max() - triangle_count)
        break;
      const size_t next_triangles = triangle_count + addition;
      if (next_triangles >
          static_cast<size_t>(std::numeric_limits<int>::max() / 3))
        break;
      if (end > begin && runtime.impl_->growth(next_triangles) > budget)
        break;
      triangle_count = next_triangles;
      ++end;
    }
    if (end == begin)
      ++end;
    run_wave(inputs, begin, end, *runtime.impl_, false);
    begin = end;
  }
}

void separate_components_batch(
    const std::vector<ComponentBatchInput> &inputs,
    ComponentBatchRuntime &runtime, size_t max_batch_size,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Component memory fraction must be in (0, 1]");
  }
  validate_inputs(inputs, true);

  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo component compaction");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
    size_t end = begin;
    size_t triangle_count = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const size_t addition = inputs[end].mesh->triangles.size();
      if (addition > std::numeric_limits<size_t>::max() - triangle_count)
        break;
      const size_t next_triangles = triangle_count + addition;
      if (next_triangles >
          static_cast<size_t>(std::numeric_limits<int>::max() / 3))
        break;
      if (end > begin && runtime.impl_->growth(next_triangles) > budget)
        break;
      triangle_count = next_triangles;
      ++end;
    }
    if (end == begin)
      ++end;
    run_wave(inputs, begin, end, *runtime.impl_, true);
    begin = end;
  }
}

} // namespace neural_acd
