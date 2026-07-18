#include <algorithm>
#include <cmath>
#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <merge_cost_batch.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace neural_acd {
namespace {

constexpr int kThreads = 256;

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

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

struct PackedProximityJob {
  const double3 *first;
  const double3 *second;
  int first_count;
  int second_count;
  double threshold_squared;
};

struct PackedVolumeJob {
  const double3 *vertices;
  const int3 *triangles;
  int triangle_count;
};

__device__ double squared_distance(double3 first, double3 second) {
  const double x = first.x - second.x;
  const double y = first.y - second.y;
  const double z = first.z - second.z;
  return x * x + y * y + z * z;
}

__global__ void proximity_kernel(const PackedProximityJob *jobs,
                                 int *results, int job_count) {
  const int job_index = blockIdx.x;
  if (job_index >= job_count)
    return;
  __shared__ int found;
  if (threadIdx.x == 0)
    found = 0;
  __syncthreads();
  const PackedProximityJob job = jobs[job_index];
  for (int first_index = threadIdx.x; first_index < job.first_count;
       first_index += blockDim.x) {
    if (atomicAdd(&found, 0))
      return;
    const double3 first = job.first[first_index];
    for (int second_index = 0; second_index < job.second_count;
         ++second_index) {
      if (atomicAdd(&found, 0))
        return;
      if (squared_distance(first, job.second[second_index]) <
          job.threshold_squared) {
        atomicExch(&found, 1);
        atomicExch(results + job_index, 1);
        return;
      }
    }
  }
}

__device__ double signed_triangle_volume(double3 first, double3 second,
                                         double3 third) {
  const double v321 = third.x * second.y * first.z;
  const double v231 = second.x * third.y * first.z;
  const double v312 = third.x * first.y * second.z;
  const double v132 = first.x * third.y * second.z;
  const double v213 = second.x * first.y * third.z;
  const double v123 = first.x * second.y * third.z;
  return (-v321 + v231 + v312 - v132 - v213 + v123) / 6.0;
}

__global__ void volume_kernel(const PackedVolumeJob *jobs, double *results,
                              int job_count) {
  const int job_index = blockIdx.x;
  if (job_index >= job_count)
    return;
  const PackedVolumeJob job = jobs[job_index];
  double local = 0.0;
  for (int triangle_index = threadIdx.x;
       triangle_index < job.triangle_count;
       triangle_index += blockDim.x) {
    const int3 triangle = job.triangles[triangle_index];
    local += signed_triangle_volume(job.vertices[triangle.x],
                                    job.vertices[triangle.y],
                                    job.vertices[triangle.z]);
  }
  __shared__ double partial[kThreads];
  partial[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[job_index] = partial[0];
}

} // namespace

struct MergeCostBatchRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer triangles;
  DeviceBuffer proximity_jobs;
  DeviceBuffer proximity_results;
  DeviceBuffer volume_jobs;
  DeviceBuffer volume_results;
  PinnedBuffer host_vertices;
  PinnedBuffer host_triangles;
  PinnedBuffer host_proximity_jobs;
  PinnedBuffer host_proximity_results;
  PinnedBuffer host_volume_jobs;
  PinnedBuffer host_volume_results;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags merge costs");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }
};

MergeCostBatchRuntime::MergeCostBatchRuntime()
    : impl_(std::make_unique<Impl>()) {}
MergeCostBatchRuntime::~MergeCostBatchRuntime() = default;
MergeCostBatchRuntime::MergeCostBatchRuntime(
    MergeCostBatchRuntime &&) noexcept = default;
MergeCostBatchRuntime &
MergeCostBatchRuntime::operator=(MergeCostBatchRuntime &&) noexcept =
    default;

namespace {

void validate_proximity_inputs(
    const std::vector<MeshProximityBatchInput> &inputs) {
  for (const MeshProximityBatchInput &input : inputs) {
    if (!input.first || !input.second || !input.within_threshold) {
      throw std::invalid_argument(
          "Mesh-proximity input contains a null pointer");
    }
    if (!std::isfinite(input.threshold) || input.threshold < 0.0) {
      throw std::invalid_argument(
          "Mesh-proximity threshold must be finite and non-negative");
    }
    if (input.first->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.second->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "Mesh-proximity input exceeds indexing limits");
    }
    if (input.first_device &&
        device_mesh_view(*input.first_device).vertex_count !=
            input.first->vertices.size()) {
      throw std::invalid_argument(
          "First proximity device mesh does not match");
    }
    if (input.second_device &&
        device_mesh_view(*input.second_device).vertex_count !=
            input.second->vertices.size()) {
      throw std::invalid_argument(
          "Second proximity device mesh does not match");
    }
  }
}

void validate_volume_inputs(
    const std::vector<MeshVolumeBatchInput> &inputs) {
  for (const MeshVolumeBatchInput &input : inputs) {
    if (!input.mesh || !input.volume)
      throw std::invalid_argument("Mesh-volume input contains a null pointer");
    if (input.mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.mesh->triangles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("Mesh-volume input exceeds indexing limits");
    }
    if (input.device_mesh) {
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      if (view.vertex_count != input.mesh->vertices.size() ||
          view.triangle_count != input.mesh->triangles.size()) {
        throw std::invalid_argument("Volume device mesh does not match");
      }
    }
    for (const auto &triangle : input.mesh->triangles) {
      for (int vertex : triangle) {
        if (vertex < 0 ||
            static_cast<size_t>(vertex) >= input.mesh->vertices.size()) {
          throw std::invalid_argument(
              "Mesh-volume input contains an invalid triangle");
        }
      }
    }
  }
}

size_t proximity_bytes(const MeshProximityBatchInput &input) {
  size_t bytes = sizeof(PackedProximityJob) + sizeof(int);
  if (!input.first_device) {
    bytes = checked_add(
        bytes,
        checked_multiply(input.first->vertices.size(), sizeof(double3),
                         "Proximity vertex bytes overflow"),
        "Proximity input bytes overflow");
  }
  if (!input.second_device) {
    bytes = checked_add(
        bytes,
        checked_multiply(input.second->vertices.size(), sizeof(double3),
                         "Proximity vertex bytes overflow"),
        "Proximity input bytes overflow");
  }
  return bytes;
}

size_t volume_bytes(const MeshVolumeBatchInput &input) {
  size_t bytes = sizeof(PackedVolumeJob) + sizeof(double);
  if (!input.device_mesh) {
    bytes = checked_add(
        bytes,
        checked_multiply(input.mesh->vertices.size(), sizeof(double3),
                         "Volume vertex bytes overflow"),
        "Volume input bytes overflow");
    bytes = checked_add(
        bytes,
        checked_multiply(input.mesh->triangles.size(), sizeof(int3),
                         "Volume triangle bytes overflow"),
        "Volume input bytes overflow");
  }
  return bytes;
}

void run_proximity_wave(
    const std::vector<MeshProximityBatchInput> &inputs, size_t begin,
    size_t end, MergeCostBatchRuntime::Impl &runtime) {
  const size_t job_count = end - begin;
  size_t vertex_count = 0;
  for (size_t index = begin; index < end; ++index) {
    if (!inputs[index].first_device) {
      vertex_count = checked_add(vertex_count,
                                 inputs[index].first->vertices.size(),
                                 "Packed proximity vertices overflow");
    }
    if (!inputs[index].second_device) {
      vertex_count = checked_add(vertex_count,
                                 inputs[index].second->vertices.size(),
                                 "Packed proximity vertices overflow");
    }
  }
  if (vertex_count >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      job_count >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Packed proximity input exceeds indexing limits");
  }

  runtime.ensure_stream();
  runtime.vertices.ensure(vertex_count * sizeof(double3),
                          "cudaMalloc proximity vertices");
  runtime.proximity_jobs.ensure(job_count * sizeof(PackedProximityJob),
                                "cudaMalloc proximity jobs");
  runtime.proximity_results.ensure(job_count * sizeof(int),
                                   "cudaMalloc proximity results");
  runtime.host_vertices.ensure(vertex_count * sizeof(double3),
                               "cudaMallocHost proximity vertices");
  runtime.host_proximity_jobs.ensure(
      job_count * sizeof(PackedProximityJob),
      "cudaMallocHost proximity jobs");
  runtime.host_proximity_results.ensure(
      job_count * sizeof(int), "cudaMallocHost proximity results");

  double3 *host_vertices = runtime.host_vertices.as<double3>();
  PackedProximityJob *host_jobs =
      runtime.host_proximity_jobs.as<PackedProximityJob>();
  size_t vertex_offset = 0;
  const auto prepare_vertices =
      [&](const Mesh &mesh, const DeviceMesh *device_mesh) {
        if (device_mesh) {
          wait_for_device_mesh(*device_mesh, runtime.stream);
          const DeviceMeshView view = device_mesh_view(*device_mesh);
          return reinterpret_cast<const double3 *>(view.vertices);
        }
        const double3 *result =
            runtime.vertices.as<double3>() + vertex_offset;
        for (const Vec3D &vertex : mesh.vertices) {
          host_vertices[vertex_offset++] =
              make_double3(vertex[0], vertex[1], vertex[2]);
        }
        return result;
      };

  for (size_t index = begin; index < end; ++index) {
    const MeshProximityBatchInput &input = inputs[index];
    host_jobs[index - begin] = {
        prepare_vertices(*input.first, input.first_device),
        prepare_vertices(*input.second, input.second_device),
        static_cast<int>(input.first->vertices.size()),
        static_cast<int>(input.second->vertices.size()),
        input.threshold * input.threshold};
  }

  if (vertex_count) {
    check_cuda(cudaMemcpyAsync(runtime.vertices.as<double3>(), host_vertices,
                               vertex_count * sizeof(double3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy proximity vertices");
  }
  check_cuda(cudaMemcpyAsync(
                 runtime.proximity_jobs.as<PackedProximityJob>(), host_jobs,
                 job_count * sizeof(PackedProximityJob),
                 cudaMemcpyHostToDevice, runtime.stream),
             "copy proximity jobs");
  check_cuda(cudaMemsetAsync(runtime.proximity_results.as<int>(), 0,
                             job_count * sizeof(int), runtime.stream),
             "clear proximity results");
  proximity_kernel<<<static_cast<int>(job_count), kThreads, 0,
                     runtime.stream>>>(
      runtime.proximity_jobs.as<PackedProximityJob>(),
      runtime.proximity_results.as<int>(), static_cast<int>(job_count));
  check_cuda(cudaGetLastError(), "launch mesh proximity");
  check_cuda(cudaMemcpyAsync(
                 runtime.host_proximity_results.as<int>(),
                 runtime.proximity_results.as<int>(),
                 job_count * sizeof(int), cudaMemcpyDeviceToHost,
                 runtime.stream),
             "copy proximity results");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "cudaStreamSynchronize proximity");
  const int *results = runtime.host_proximity_results.as<int>();
  for (size_t index = begin; index < end; ++index)
    *inputs[index].within_threshold = results[index - begin] != 0;
}

void run_volume_wave(const std::vector<MeshVolumeBatchInput> &inputs,
                     size_t begin, size_t end,
                     MergeCostBatchRuntime::Impl &runtime) {
  const size_t job_count = end - begin;
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  for (size_t index = begin; index < end; ++index) {
    if (inputs[index].device_mesh)
      continue;
    vertex_count = checked_add(vertex_count,
                               inputs[index].mesh->vertices.size(),
                               "Packed volume vertices overflow");
    triangle_count = checked_add(triangle_count,
                                 inputs[index].mesh->triangles.size(),
                                 "Packed volume triangles overflow");
  }
  if (vertex_count >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      triangle_count >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      job_count >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Packed volume input exceeds indexing limits");
  }

  runtime.ensure_stream();
  runtime.vertices.ensure(vertex_count * sizeof(double3),
                          "cudaMalloc volume vertices");
  runtime.triangles.ensure(triangle_count * sizeof(int3),
                           "cudaMalloc volume triangles");
  runtime.volume_jobs.ensure(job_count * sizeof(PackedVolumeJob),
                             "cudaMalloc volume jobs");
  runtime.volume_results.ensure(job_count * sizeof(double),
                                "cudaMalloc volume results");
  runtime.host_vertices.ensure(vertex_count * sizeof(double3),
                               "cudaMallocHost volume vertices");
  runtime.host_triangles.ensure(triangle_count * sizeof(int3),
                                "cudaMallocHost volume triangles");
  runtime.host_volume_jobs.ensure(job_count * sizeof(PackedVolumeJob),
                                  "cudaMallocHost volume jobs");
  runtime.host_volume_results.ensure(job_count * sizeof(double),
                                     "cudaMallocHost volume results");

  double3 *host_vertices = runtime.host_vertices.as<double3>();
  int3 *host_triangles = runtime.host_triangles.as<int3>();
  PackedVolumeJob *host_jobs =
      runtime.host_volume_jobs.as<PackedVolumeJob>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  for (size_t index = begin; index < end; ++index) {
    const MeshVolumeBatchInput &input = inputs[index];
    const double3 *vertices = nullptr;
    const int3 *triangles = nullptr;
    if (input.device_mesh) {
      wait_for_device_mesh(*input.device_mesh, runtime.stream);
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      vertices = reinterpret_cast<const double3 *>(view.vertices);
      triangles = reinterpret_cast<const int3 *>(view.triangles);
    } else {
      vertices = runtime.vertices.as<double3>() + vertex_offset;
      triangles = runtime.triangles.as<int3>() + triangle_offset;
      for (const Vec3D &vertex : input.mesh->vertices) {
        host_vertices[vertex_offset++] =
            make_double3(vertex[0], vertex[1], vertex[2]);
      }
      for (const auto &triangle : input.mesh->triangles) {
        host_triangles[triangle_offset++] =
            make_int3(triangle[0], triangle[1], triangle[2]);
      }
    }
    host_jobs[index - begin] = {
        vertices, triangles,
        static_cast<int>(input.mesh->triangles.size())};
  }

  if (vertex_count) {
    check_cuda(cudaMemcpyAsync(runtime.vertices.as<double3>(), host_vertices,
                               vertex_count * sizeof(double3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy volume vertices");
  }
  if (triangle_count) {
    check_cuda(cudaMemcpyAsync(runtime.triangles.as<int3>(), host_triangles,
                               triangle_count * sizeof(int3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy volume triangles");
  }
  check_cuda(cudaMemcpyAsync(runtime.volume_jobs.as<PackedVolumeJob>(),
                             host_jobs,
                             job_count * sizeof(PackedVolumeJob),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy volume jobs");
  volume_kernel<<<static_cast<int>(job_count), kThreads, 0,
                  runtime.stream>>>(
      runtime.volume_jobs.as<PackedVolumeJob>(),
      runtime.volume_results.as<double>(), static_cast<int>(job_count));
  check_cuda(cudaGetLastError(), "launch mesh volumes");
  check_cuda(cudaMemcpyAsync(runtime.host_volume_results.as<double>(),
                             runtime.volume_results.as<double>(),
                             job_count * sizeof(double),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy volume results");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "cudaStreamSynchronize volumes");
  const double *results = runtime.host_volume_results.as<double>();
  for (size_t index = begin; index < end; ++index)
    *inputs[index].volume = results[index - begin];
}

template <typename Input, typename Bytes, typename RunWave>
void run_waves(const std::vector<Input> &inputs, size_t max_batch_size,
               double memory_fraction, Bytes bytes_for_input,
               RunWave run_wave) {
  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo merge costs");
    const size_t budget =
        std::max<size_t>(1, static_cast<size_t>(
                                free_bytes * memory_fraction));
    const size_t hard_end =
        max_batch_size == 0
            ? inputs.size()
            : std::min(inputs.size(), begin + max_batch_size);
    size_t end = begin;
    size_t bytes = 0;
    while (end < hard_end) {
      const size_t next = bytes_for_input(inputs[end]);
      if (end > begin && next > budget - std::min(bytes, budget))
        break;
      bytes = checked_add(bytes, next, "Merge-cost wave bytes overflow");
      ++end;
    }
    if (end == begin)
      ++end;
    run_wave(begin, end);
    begin = end;
  }
}

} // namespace

void evaluate_mesh_proximity_batch(
    const std::vector<MeshProximityBatchInput> &inputs,
    MergeCostBatchRuntime &runtime, size_t max_batch_size,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Merge-cost memory fraction must be in (0, 1]");
  }
  validate_proximity_inputs(inputs);
  run_waves(
      inputs, max_batch_size, memory_fraction, proximity_bytes,
      [&](size_t begin, size_t end) {
        run_proximity_wave(inputs, begin, end, *runtime.impl_);
      });
}

void evaluate_mesh_volumes_batch(
    const std::vector<MeshVolumeBatchInput> &inputs,
    MergeCostBatchRuntime &runtime, size_t max_batch_size,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Merge-cost memory fraction must be in (0, 1]");
  }
  validate_volume_inputs(inputs);
  run_waves(
      inputs, max_batch_size, memory_fraction, volume_bytes,
      [&](size_t begin, size_t end) {
        run_volume_wave(inputs, begin, end, *runtime.impl_);
      });
}

} // namespace neural_acd
