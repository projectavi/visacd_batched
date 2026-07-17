#include <cuda_runtime.h>
#include <device_mesh.hpp>
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

size_t checked_add(size_t first, size_t second, const char *message) {
  if (second > std::numeric_limits<size_t>::max() - first)
    throw std::overflow_error(message);
  return first + second;
}

size_t checked_multiply(size_t first, size_t second, const char *message) {
  if (first != 0 && second > std::numeric_limits<size_t>::max() / first)
    throw std::overflow_error(message);
  return first * second;
}

size_t mesh_bytes(const Mesh &mesh) {
  size_t bytes = checked_multiply(mesh.vertices.size(),
                                  sizeof(double3) + sizeof(float3),
                                  "Device mesh vertex bytes overflow");
  bytes = checked_add(
      bytes,
      checked_multiply(mesh.triangles.size(), sizeof(int3),
                       "Device mesh triangle bytes overflow"),
      "Device mesh bytes overflow");
  return checked_add(
      bytes,
      checked_multiply(mesh.intersecting_edges.size(), sizeof(uint2),
                       "Device mesh edge bytes overflow"),
      "Device mesh bytes overflow");
}

template <typename T> class DeviceArray {
public:
  DeviceArray() = default;
  ~DeviceArray() {
    if (data_)
      cudaFree(data_);
  }

  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;

  void allocate(size_t count, const char *operation) {
    if (count == 0)
      return;
    check_cuda(cudaMalloc(&data_, checked_multiply(count, sizeof(T),
                                                   "Device array overflow")),
               operation);
  }

  T *get() const { return data_; }

private:
  T *data_ = nullptr;
};

__global__ void convert_vertices_kernel(const double3 *input, float3 *output,
                                        size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const double3 vertex = input[index];
  output[index] = make_float3(static_cast<float>(vertex.x),
                              static_cast<float>(vertex.y),
                              static_cast<float>(vertex.z));
}

} // namespace

struct DeviceMesh::Impl {
  DeviceArray<double3> vertices;
  DeviceArray<float3> float_vertices;
  DeviceArray<int3> triangles;
  DeviceArray<uint2> edges;
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  size_t edge_count = 0;
  std::vector<uint2> staging_edges;
  cudaEvent_t ready_event = nullptr;

  ~Impl() {
    if (ready_event) {
      cudaEventSynchronize(ready_event);
      cudaEventDestroy(ready_event);
    }
  }
};

struct DeviceMeshRuntime::Impl {
  cudaStream_t stream = nullptr;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags device mesh upload");
    }
  }
};

DeviceMesh::DeviceMesh() : impl_(std::make_unique<Impl>()) {}
DeviceMesh::~DeviceMesh() = default;
DeviceMesh::DeviceMesh(DeviceMesh &&) noexcept = default;
DeviceMesh &DeviceMesh::operator=(DeviceMesh &&) noexcept = default;

DeviceMeshRuntime::DeviceMeshRuntime() : impl_(std::make_unique<Impl>()) {}
DeviceMeshRuntime::~DeviceMeshRuntime() = default;
DeviceMeshRuntime::DeviceMeshRuntime(DeviceMeshRuntime &&) noexcept = default;
DeviceMeshRuntime &
DeviceMeshRuntime::operator=(DeviceMeshRuntime &&) noexcept = default;

std::shared_ptr<DeviceMesh>
DeviceMeshRuntime::try_upload(const Mesh &mesh, double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument("Device mesh memory fraction must be in (0, 1]");
  }
  if (mesh.vertices.empty() || mesh.triangles.empty())
    return nullptr;
  if (mesh.vertices.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      mesh.triangles.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      mesh.intersecting_edges.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
             "cudaMemGetInfo device mesh upload");
  const size_t required = mesh_bytes(mesh);
  const size_t budget =
      static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
  if (required > budget)
    return nullptr;

  impl_->ensure_stream();
  auto result = std::make_shared<DeviceMesh>();
  DeviceMesh::Impl &device = *result->impl_;
  device.vertex_count = mesh.vertices.size();
  device.triangle_count = mesh.triangles.size();
  device.edge_count = mesh.intersecting_edges.size();
  device.vertices.allocate(device.vertex_count,
                           "cudaMalloc device mesh vertices");
  device.float_vertices.allocate(device.vertex_count,
                                 "cudaMalloc device mesh float vertices");
  device.triangles.allocate(device.triangle_count,
                            "cudaMalloc device mesh triangles");
  device.edges.allocate(device.edge_count, "cudaMalloc device mesh edges");
  check_cuda(cudaEventCreateWithFlags(&device.ready_event,
                                      cudaEventDisableTiming),
             "cudaEventCreate device mesh");

  check_cuda(cudaMemcpyAsync(device.vertices.get(), mesh.vertices.data(),
                             device.vertex_count * sizeof(double3),
                             cudaMemcpyHostToDevice, impl_->stream),
             "copy device mesh vertices");
  check_cuda(cudaMemcpyAsync(device.triangles.get(), mesh.triangles.data(),
                             device.triangle_count * sizeof(int3),
                             cudaMemcpyHostToDevice, impl_->stream),
             "copy device mesh triangles");
  if (device.edge_count) {
    device.staging_edges.reserve(device.edge_count);
    for (const auto &edge : mesh.intersecting_edges)
      device.staging_edges.push_back(make_uint2(edge.first, edge.second));
    check_cuda(cudaMemcpyAsync(device.edges.get(), device.staging_edges.data(),
                               device.edge_count * sizeof(uint2),
                               cudaMemcpyHostToDevice, impl_->stream),
               "copy device mesh edges");
  }

  constexpr int block_size = 256;
  const int blocks = static_cast<int>(
      (device.vertex_count + block_size - 1) / block_size);
  convert_vertices_kernel<<<blocks, block_size, 0, impl_->stream>>>(
      device.vertices.get(), device.float_vertices.get(),
      device.vertex_count);
  check_cuda(cudaGetLastError(), "launch device mesh vertex conversion");
  check_cuda(cudaEventRecord(device.ready_event, impl_->stream),
             "cudaEventRecord device mesh");
  return result;
}

DeviceMeshView device_mesh_view(const DeviceMesh &mesh) {
  const DeviceMesh::Impl &device = *mesh.impl_;
  return {reinterpret_cast<const double *>(device.vertices.get()),
          reinterpret_cast<const float *>(device.float_vertices.get()),
          reinterpret_cast<const int *>(device.triangles.get()),
          reinterpret_cast<const unsigned int *>(device.edges.get()),
          device.vertex_count,
          device.triangle_count,
          device.edge_count,
          device.ready_event};
}

void wait_for_device_mesh(const DeviceMesh &mesh, void *stream) {
  const DeviceMeshView view = device_mesh_view(mesh);
  check_cuda(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                                 static_cast<cudaEvent_t>(view.ready_event)),
             "cudaStreamWaitEvent device mesh");
}

} // namespace neural_acd
