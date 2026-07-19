#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <device_mesh.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
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

using cuda_memory::DeviceBuffer;

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

__global__ void import_meshed_vertices_kernel(
    const float3 *input, double3 *vertices, float3 *float_vertices,
    size_t count, double scale) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const float3 point = input[index];
  const double3 vertex =
      make_double3(static_cast<double>(point.x) / scale,
                   static_cast<double>(point.y) / scale,
                   static_cast<double>(point.z) / scale);
  vertices[index] = vertex;
  float_vertices[index] =
      make_float3(static_cast<float>(vertex.x),
                  static_cast<float>(vertex.y),
                  static_cast<float>(vertex.z));
}

__global__ void import_meshed_quads_kernel(const int4 *quads,
                                           int3 *triangles,
                                           size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const int4 quad = quads[index];
  triangles[index * 2] = make_int3(quad.x, quad.z, quad.y);
  triangles[index * 2 + 1] = make_int3(quad.x, quad.w, quad.z);
}

__global__ void remap_device_mesh_vertices_kernel(
    const double3 *source_vertices, const float3 *source_float_vertices,
    const int *source_indices, double3 *vertices, float3 *float_vertices,
    size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const int source = source_indices[index];
  vertices[index] = source_vertices[source];
  float_vertices[index] = source_float_vertices[source];
}

} // namespace

struct DeviceMesh::Impl {
  DeviceBuffer vertices;
  DeviceBuffer float_vertices;
  DeviceBuffer triangles;
  DeviceBuffer edges;
  DeviceBuffer remap_indices;
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
  std::mutex mutex;
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
    DeviceBuffer::set_allocation_stream(stream);
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

  std::lock_guard<std::mutex> lock(impl_->mutex);

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
  device.vertices.ensure(
      checked_multiply(device.vertex_count, sizeof(double3),
                       "Device mesh vertex allocation overflow"),
      "allocate device mesh vertices");
  device.float_vertices.ensure(
      checked_multiply(device.vertex_count, sizeof(float3),
                       "Device mesh float vertex allocation overflow"),
      "allocate device mesh float vertices");
  device.triangles.ensure(
      checked_multiply(device.triangle_count, sizeof(int3),
                       "Device mesh triangle allocation overflow"),
      "allocate device mesh triangles");
  if (device.edge_count) {
    device.edges.ensure(
        checked_multiply(device.edge_count, sizeof(uint2),
                         "Device mesh edge allocation overflow"),
        "allocate device mesh edges");
  }
  check_cuda(cudaEventCreateWithFlags(&device.ready_event,
                                      cudaEventDisableTiming),
             "cudaEventCreate device mesh");

  check_cuda(cudaMemcpyAsync(device.vertices.as<double3>(), mesh.vertices.data(),
                             device.vertex_count * sizeof(double3),
                             cudaMemcpyHostToDevice, impl_->stream),
             "copy device mesh vertices");
  check_cuda(cudaMemcpyAsync(device.triangles.as<int3>(), mesh.triangles.data(),
                             device.triangle_count * sizeof(int3),
                             cudaMemcpyHostToDevice, impl_->stream),
             "copy device mesh triangles");
  if (device.edge_count) {
    device.staging_edges.reserve(device.edge_count);
    for (const auto &edge : mesh.intersecting_edges)
      device.staging_edges.push_back(make_uint2(edge.first, edge.second));
    check_cuda(cudaMemcpyAsync(
                   device.edges.as<uint2>(), device.staging_edges.data(),
                   device.edge_count * sizeof(uint2), cudaMemcpyHostToDevice,
                   impl_->stream),
               "copy device mesh edges");
  }

  constexpr int block_size = 256;
  const int blocks = static_cast<int>(
      (device.vertex_count + block_size - 1) / block_size);
  convert_vertices_kernel<<<blocks, block_size, 0, impl_->stream>>>(
      device.vertices.as<double3>(), device.float_vertices.as<float3>(),
      device.vertex_count);
  check_cuda(cudaGetLastError(), "launch device mesh vertex conversion");
  check_cuda(cudaEventRecord(device.ready_event, impl_->stream),
             "cudaEventRecord device mesh");
  return result;
}

bool DeviceMeshRuntime::try_attach_edges(
    const std::shared_ptr<DeviceMesh> &device_mesh, const Mesh &mesh,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Device mesh memory fraction must be in (0, 1]");
  }
  if (!device_mesh)
    return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  DeviceMesh::Impl &device = *device_mesh->impl_;
  if (device.vertex_count != mesh.vertices.size() ||
      device.triangle_count != mesh.triangles.size() ||
      mesh.intersecting_edges.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  if (mesh.intersecting_edges.empty()) {
    device.edge_count = 0;
    return true;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
             "cudaMemGetInfo device mesh edges");
  const size_t required = checked_multiply(
      mesh.intersecting_edges.size(), sizeof(uint2),
      "Device mesh edge bytes overflow");
  const size_t budget =
      static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
  if (required > budget)
    return false;

  impl_->ensure_stream();
  if (device.ready_event) {
    check_cuda(cudaStreamWaitEvent(impl_->stream, device.ready_event, 0),
               "cudaStreamWaitEvent device mesh edges");
  }
  device.edges.ensure(required, "allocate retained device mesh edges");
  device.staging_edges.clear();
  device.staging_edges.reserve(mesh.intersecting_edges.size());
  for (const auto &edge : mesh.intersecting_edges)
    device.staging_edges.push_back(make_uint2(edge.first, edge.second));
  check_cuda(cudaMemcpyAsync(
                 device.edges.as<uint2>(), device.staging_edges.data(),
                 required, cudaMemcpyHostToDevice, impl_->stream),
             "copy retained device mesh edges");
  device.edge_count = mesh.intersecting_edges.size();
  check_cuda(cudaEventRecord(device.ready_event, impl_->stream),
             "cudaEventRecord retained device mesh edges");
  return true;
}

std::shared_ptr<DeviceMesh> DeviceMeshRuntime::try_remap(
    const std::shared_ptr<DeviceMesh> &source, const Mesh &mesh,
    const std::vector<int> &source_vertices, double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Device mesh memory fraction must be in (0, 1]");
  }
  if (!source || mesh.vertices.empty() || mesh.triangles.empty() ||
      source_vertices.size() != mesh.vertices.size()) {
    return nullptr;
  }
  const DeviceMeshView source_view = device_mesh_view(*source);
  if (mesh.vertices.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      mesh.triangles.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      mesh.intersecting_edges.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }
  for (int vertex : source_vertices) {
    if (vertex < 0 || static_cast<size_t>(vertex) >= source_view.vertex_count)
      return nullptr;
  }

  std::lock_guard<std::mutex> lock(impl_->mutex);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
             "cudaMemGetInfo remapped device mesh");
  size_t required = mesh_bytes(mesh);
  required = checked_add(
      required,
      checked_multiply(source_vertices.size(), sizeof(int),
                       "Device mesh remap bytes overflow"),
      "Device mesh remap bytes overflow");
  const size_t budget =
      static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
  if (required > budget)
    return nullptr;

  impl_->ensure_stream();
  wait_for_device_mesh(*source, impl_->stream);
  auto result = std::make_shared<DeviceMesh>();
  DeviceMesh::Impl &device = *result->impl_;
  device.vertex_count = mesh.vertices.size();
  device.triangle_count = mesh.triangles.size();
  device.edge_count = mesh.intersecting_edges.size();
  device.vertices.ensure(device.vertex_count * sizeof(double3),
                         "allocate remapped device mesh vertices");
  device.float_vertices.ensure(
      device.vertex_count * sizeof(float3),
      "allocate remapped device mesh float vertices");
  device.triangles.ensure(device.triangle_count * sizeof(int3),
                          "allocate remapped device mesh triangles");
  device.remap_indices.ensure(
      device.vertex_count * sizeof(int),
      "allocate remapped device mesh indices");
  if (device.edge_count) {
    device.edges.ensure(device.edge_count * sizeof(uint2),
                        "allocate remapped device mesh edges");
  }
  check_cuda(cudaEventCreateWithFlags(&device.ready_event,
                                      cudaEventDisableTiming),
             "cudaEventCreate remapped device mesh");
  check_cuda(cudaMemcpyAsync(
                 device.remap_indices.as<int>(), source_vertices.data(),
                 device.vertex_count * sizeof(int), cudaMemcpyHostToDevice,
                 impl_->stream),
             "copy remapped device mesh indices");
  constexpr int block_size = 256;
  const int blocks = static_cast<int>(
      (device.vertex_count + block_size - 1) / block_size);
  remap_device_mesh_vertices_kernel<<<blocks, block_size, 0,
                                      impl_->stream>>>(
      reinterpret_cast<const double3 *>(source_view.vertices),
      reinterpret_cast<const float3 *>(source_view.float_vertices),
      device.remap_indices.as<int>(), device.vertices.as<double3>(),
      device.float_vertices.as<float3>(), device.vertex_count);
  check_cuda(cudaGetLastError(),
             "launch remapped device mesh vertex gather");
  check_cuda(cudaMemcpyAsync(
                 device.triangles.as<int3>(), mesh.triangles.data(),
                 device.triangle_count * sizeof(int3),
                 cudaMemcpyHostToDevice, impl_->stream),
             "copy remapped device mesh triangles");
  if (device.edge_count) {
    device.staging_edges.reserve(device.edge_count);
    for (const auto &edge : mesh.intersecting_edges)
      device.staging_edges.push_back(make_uint2(edge.first, edge.second));
    check_cuda(cudaMemcpyAsync(
                   device.edges.as<uint2>(), device.staging_edges.data(),
                   device.edge_count * sizeof(uint2),
                   cudaMemcpyHostToDevice, impl_->stream),
               "copy remapped device mesh edges");
  }
  check_cuda(cudaEventRecord(device.ready_event, impl_->stream),
             "cudaEventRecord remapped device mesh");
  return result;
}

std::shared_ptr<DeviceMesh> try_make_device_mesh_from_quads(
    const float *device_points, size_t point_count,
    const int *device_quads, size_t quad_count, double scale,
    void *producer_stream, double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Device mesh memory fraction must be in (0, 1]");
  }
  if (!std::isfinite(scale) || scale <= 0.0)
    throw std::invalid_argument("Device mesh import scale must be positive");
  if (!device_points || !device_quads || !producer_stream ||
      point_count == 0 || quad_count == 0) {
    return nullptr;
  }
  if (point_count >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      quad_count >
          static_cast<size_t>(std::numeric_limits<int>::max() / 2)) {
    return nullptr;
  }
  const size_t triangle_count = checked_multiply(
      quad_count, size_t(2), "Device mesh triangle count overflow");
  size_t required = checked_multiply(
      point_count, sizeof(double3) + sizeof(float3),
      "Device mesh imported vertex bytes overflow");
  required = checked_add(
      required,
      checked_multiply(triangle_count, sizeof(int3),
                       "Device mesh imported triangle bytes overflow"),
      "Device mesh imported bytes overflow");
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
             "cudaMemGetInfo device mesh import");
  const size_t budget =
      static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
  if (required > budget)
    return nullptr;

  const cudaStream_t stream =
      static_cast<cudaStream_t>(producer_stream);
  DeviceBuffer::set_allocation_stream(stream);
  auto result = std::make_shared<DeviceMesh>();
  DeviceMesh::Impl &device = *result->impl_;
  device.vertex_count = point_count;
  device.triangle_count = triangle_count;
  device.vertices.ensure(point_count * sizeof(double3),
                         "allocate imported device mesh vertices");
  device.float_vertices.ensure(
      point_count * sizeof(float3),
      "allocate imported device mesh float vertices");
  device.triangles.ensure(triangle_count * sizeof(int3),
                          "allocate imported device mesh triangles");
  check_cuda(cudaEventCreateWithFlags(&device.ready_event,
                                      cudaEventDisableTiming),
             "cudaEventCreate imported device mesh");

  constexpr int block_size = 256;
  const int point_blocks = static_cast<int>(
      (point_count + block_size - 1) / block_size);
  import_meshed_vertices_kernel<<<point_blocks, block_size, 0, stream>>>(
      reinterpret_cast<const float3 *>(device_points),
      device.vertices.as<double3>(), device.float_vertices.as<float3>(),
      point_count, scale);
  check_cuda(cudaGetLastError(),
             "launch imported device mesh vertex conversion");
  const int quad_blocks = static_cast<int>(
      (quad_count + block_size - 1) / block_size);
  import_meshed_quads_kernel<<<quad_blocks, block_size, 0, stream>>>(
      reinterpret_cast<const int4 *>(device_quads),
      device.triangles.as<int3>(), quad_count);
  check_cuda(cudaGetLastError(),
             "launch imported device mesh quad triangulation");
  check_cuda(cudaEventRecord(device.ready_event, stream),
             "cudaEventRecord imported device mesh");
  return result;
}

DeviceMeshView device_mesh_view(const DeviceMesh &mesh) {
  const DeviceMesh::Impl &device = *mesh.impl_;
  return {reinterpret_cast<const double *>(device.vertices.as<double3>()),
          reinterpret_cast<const float *>(device.float_vertices.as<float3>()),
          reinterpret_cast<const int *>(device.triangles.as<int3>()),
          reinterpret_cast<const unsigned int *>(device.edges.as<uint2>()),
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
