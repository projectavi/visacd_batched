#include <algorithm>
#include <components_batch.hpp>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
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

class DeviceBuffer {
public:
  ~DeviceBuffer() {
    if (data_)
      cudaFree(data_);
  }

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void ensure(size_t bytes, const char *operation) {
    if (bytes <= capacity_)
      return;
    if (data_)
      check_cuda(cudaFree(data_), "cudaFree component buffer");
    data_ = nullptr;
    capacity_ = 0;
    check_cuda(cudaMalloc(&data_, bytes), operation);
    capacity_ = bytes;
  }

  template <typename T> T *as() const { return static_cast<T *>(data_); }
  size_t capacity() const { return capacity_; }

private:
  void *data_ = nullptr;
  size_t capacity_ = 0;
};

class PinnedBuffer {
public:
  ~PinnedBuffer() {
    if (data_)
      cudaFreeHost(data_);
  }

  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  void ensure(size_t bytes, const char *operation) {
    if (bytes <= capacity_)
      return;
    if (data_)
      check_cuda(cudaFreeHost(data_), "cudaFreeHost component buffer");
    data_ = nullptr;
    capacity_ = 0;
    check_cuda(cudaMallocHost(&data_, bytes), operation);
    capacity_ = bytes;
  }

  template <typename T> T *as() const { return static_cast<T *>(data_); }

private:
  void *data_ = nullptr;
  size_t capacity_ = 0;
};

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
  DeviceBuffer sort_temp;
  PinnedBuffer host_triangles;
  PinnedBuffer host_labels;

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

void validate_inputs(const std::vector<ComponentBatchInput> &inputs) {
  for (const ComponentBatchInput &input : inputs) {
    if (!input.mesh || !input.labels)
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
              size_t end, ComponentBatchRuntime::Impl &runtime) {
  size_t vertex_count = 0;
  size_t triangle_count = 0;
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
  }
  if (triangle_count == 0) {
    for (size_t index = begin; index < end; ++index)
      inputs[index].labels->clear();
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

  int3 *host_triangles = runtime.host_triangles.as<int3>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  for (size_t index = begin; index < end; ++index) {
    const Mesh &mesh = *inputs[index].mesh;
    for (const auto &triangle : mesh.triangles) {
      host_triangles[triangle_offset++] = make_int3(
          static_cast<int>(vertex_offset) + triangle[0],
          static_cast<int>(vertex_offset) + triangle[1],
          static_cast<int>(vertex_offset) + triangle[2]);
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
  validate_inputs(inputs);

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
    run_wave(inputs, begin, end, *runtime.impl_);
    begin = end;
  }
}

} // namespace neural_acd
