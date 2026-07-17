#include <algorithm>
#include <clip_batch.hpp>
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
      check_cuda(cudaFree(data_), "cudaFree clip buffer");
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
      check_cuda(cudaFreeHost(data_), "cudaFreeHost clip buffer");
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

struct PackedClipJob {
  const double3 *vertices;
  const int3 *triangles;
  double4 plane;
  int output_offset;
  int triangle_count;
};

__device__ double add_rn(double first, double second) {
  return __dadd_rn(first, second);
}

__device__ double subtract_rn(double first, double second) {
  return __dadd_rn(first, -second);
}

__device__ double multiply_rn(double first, double second) {
  return __dmul_rn(first, second);
}

__device__ double plane_value(double3 point, double4 plane) {
  double value = multiply_rn(point.x, plane.x);
  value = add_rn(value, multiply_rn(point.y, plane.y));
  value = add_rn(value, multiply_rn(point.z, plane.z));
  return add_rn(value, plane.w);
}

__device__ short point_side(double3 point, double4 plane) {
  const double value = plane_value(point, plane);
  if (value > 1e-6)
    return 1;
  if (value < -1e-6)
    return -1;
  return 0;
}

__device__ short coplanar_side(double3 first, double3 second, double3 third,
                               double4 plane) {
  const double3 first_edge =
      make_double3(second.x - first.x, second.y - first.y,
                   second.z - first.z);
  const double3 second_edge =
      make_double3(third.x - first.x, third.y - first.y,
                   third.z - first.z);
  const double nx = subtract_rn(multiply_rn(first_edge.y, second_edge.z),
                                multiply_rn(first_edge.z, second_edge.y));
  const double ny = subtract_rn(multiply_rn(first_edge.z, second_edge.x),
                                multiply_rn(first_edge.x, second_edge.z));
  const double nz = subtract_rn(multiply_rn(first_edge.x, second_edge.y),
                                multiply_rn(first_edge.y, second_edge.x));
  double length_squared = multiply_rn(nx, nx);
  length_squared = add_rn(length_squared, multiply_rn(ny, ny));
  length_squared = add_rn(length_squared, multiply_rn(nz, nz));
  const double length = sqrt(length_squared);
  const double normal_x = __ddiv_rn(nx, length);
  const double normal_y = __ddiv_rn(ny, length);
  const double normal_z = __ddiv_rn(nz, length);
  if (multiply_rn(normal_x, plane.x) > 0.0 ||
      multiply_rn(normal_y, plane.y) > 0.0 ||
      multiply_rn(normal_z, plane.z) > 0.0) {
    return -1;
  }
  return 1;
}

__device__ double segment_denominator(double3 first, double3 second,
                                      double4 plane) {
  double denominator = multiply_rn(plane.x, second.x);
  denominator = subtract_rn(denominator, multiply_rn(plane.x, first.x));
  denominator = add_rn(denominator, multiply_rn(plane.y, second.y));
  denominator = subtract_rn(denominator, multiply_rn(plane.y, first.y));
  denominator = add_rn(denominator, multiply_rn(plane.z, second.z));
  return subtract_rn(denominator, multiply_rn(plane.z, first.z));
}

__device__ double multiply_three(double first, double second, double third) {
  return multiply_rn(multiply_rn(first, second), third);
}

__device__ bool segment_intersection(double3 first, double3 second,
                                     double4 plane, double3 &intersection) {
  const double denominator = segment_denominator(first, second, plane);

  double x = multiply_three(first.x, plane.y, second.y);
  x = add_rn(x, multiply_three(first.x, plane.z, second.z));
  x = add_rn(x, multiply_rn(first.x, plane.w));
  x = subtract_rn(x, multiply_three(second.x, plane.y, first.y));
  x = subtract_rn(x, multiply_three(second.x, plane.z, first.z));
  x = subtract_rn(x, multiply_rn(second.x, plane.w));

  double y = multiply_three(plane.x, second.x, first.y);
  y = add_rn(y, multiply_three(plane.z, first.y, second.z));
  y = add_rn(y, multiply_rn(first.y, plane.w));
  y = subtract_rn(y, multiply_three(plane.x, first.x, second.y));
  y = subtract_rn(y, multiply_three(plane.z, first.z, second.y));
  y = subtract_rn(y, multiply_rn(second.y, plane.w));

  double z = multiply_three(plane.x, second.x, first.z);
  z = add_rn(z, multiply_three(plane.y, second.y, first.z));
  z = add_rn(z, multiply_rn(first.z, plane.w));
  z = subtract_rn(z, multiply_three(plane.x, first.x, second.z));
  z = subtract_rn(z, multiply_three(plane.y, first.y, second.z));
  z = subtract_rn(z, multiply_rn(second.z, plane.w));

  intersection = make_double3(__ddiv_rn(x, denominator),
                              __ddiv_rn(y, denominator),
                              __ddiv_rn(z, denominator));
  constexpr double eps = 1e-6;
  return fmin(first.x - eps, second.x - eps) <= intersection.x &&
         intersection.x <= fmax(first.x + eps, second.x + eps) &&
         fmin(first.y - eps, second.y - eps) <= intersection.y &&
         intersection.y <= fmax(first.y + eps, second.y + eps) &&
         fmin(first.z - eps, second.z - eps) <= intersection.z &&
         intersection.z <= fmax(first.z + eps, second.z + eps);
}

__device__ void store_intersection(ClipTriangleData &output, int edge,
                                   double3 point) {
  output.intersections[edge * 3] = point.x;
  output.intersections[edge * 3 + 1] = point.y;
  output.intersections[edge * 3 + 2] = point.z;
}

__global__ void prepare_clip_kernel(const PackedClipJob *jobs,
                                    ClipTriangleData *outputs,
                                    int job_count) {
  const int job_index = blockIdx.x;
  if (job_index >= job_count)
    return;
  const PackedClipJob job = jobs[job_index];
  for (int local_triangle = threadIdx.x;
       local_triangle < job.triangle_count;
       local_triangle += blockDim.x) {
    const int3 triangle = job.triangles[local_triangle];
    const double3 points[3] = {job.vertices[triangle.x],
                               job.vertices[triangle.y],
                               job.vertices[triangle.z]};
    ClipTriangleData output;
    output.sides[0] = point_side(points[0], job.plane);
    output.sides[1] = point_side(points[1], job.plane);
    output.sides[2] = point_side(points[2], job.plane);
    if (output.sides[0] == 0 && output.sides[1] == 0 &&
        output.sides[2] == 0) {
      const short side =
          coplanar_side(points[0], points[1], points[2], job.plane);
      output.sides[0] = side;
      output.sides[1] = side;
      output.sides[2] = side;
    }
    double3 intersection;
    if (segment_intersection(points[0], points[1], job.plane,
                             intersection)) {
      output.intersection_mask |= 1u;
      store_intersection(output, 0, intersection);
    }
    if (segment_intersection(points[1], points[2], job.plane,
                             intersection)) {
      output.intersection_mask |= 2u;
      store_intersection(output, 1, intersection);
    }
    if (segment_intersection(points[2], points[0], job.plane,
                             intersection)) {
      output.intersection_mask |= 4u;
      store_intersection(output, 2, intersection);
    }
    outputs[job.output_offset + local_triangle] = output;
  }
}

size_t growth_bytes(const DeviceBuffer &buffer, size_t bytes) {
  return bytes > buffer.capacity() ? bytes - buffer.capacity() : 0;
}

} // namespace

struct ClipBatchRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer jobs;
  DeviceBuffer outputs;
  PinnedBuffer host_jobs;
  PinnedBuffer host_outputs;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags clip batch");
    }
  }

  size_t growth(size_t jobs_count, size_t triangle_count) const {
    return growth_bytes(jobs, jobs_count * sizeof(PackedClipJob)) +
           growth_bytes(outputs, triangle_count * sizeof(ClipTriangleData));
  }
};

ClipBatchRuntime::ClipBatchRuntime() : impl_(std::make_unique<Impl>()) {}
ClipBatchRuntime::~ClipBatchRuntime() = default;
ClipBatchRuntime::ClipBatchRuntime(ClipBatchRuntime &&) noexcept = default;
ClipBatchRuntime &
ClipBatchRuntime::operator=(ClipBatchRuntime &&) noexcept = default;

namespace {

void validate_inputs(const std::vector<ClipBatchInput> &inputs) {
  for (const ClipBatchInput &input : inputs) {
    if (!input.device_mesh || !input.output)
      throw std::invalid_argument("Clip batch input contains a null pointer");
    const DeviceMeshView view = device_mesh_view(*input.device_mesh);
    if (view.triangle_count >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("Clip triangle count exceeds indexing limits");
    }
  }
}

void run_wave(const std::vector<ClipBatchInput> &inputs, size_t begin,
              size_t end, ClipBatchRuntime::Impl &runtime) {
  size_t triangle_count = 0;
  for (size_t index = begin; index < end; ++index) {
    const DeviceMeshView view = device_mesh_view(*inputs[index].device_mesh);
    if (view.triangle_count >
        static_cast<size_t>(std::numeric_limits<int>::max()) - triangle_count) {
      throw std::overflow_error("Packed clip output exceeds indexing limits");
    }
    triangle_count += view.triangle_count;
  }
  if (triangle_count == 0)
    return;

  const size_t job_count = end - begin;
  runtime.ensure_stream();
  runtime.jobs.ensure(job_count * sizeof(PackedClipJob),
                      "cudaMalloc clip jobs");
  runtime.outputs.ensure(triangle_count * sizeof(ClipTriangleData),
                         "cudaMalloc clip outputs");
  runtime.host_jobs.ensure(job_count * sizeof(PackedClipJob),
                           "cudaMallocHost clip jobs");
  runtime.host_outputs.ensure(triangle_count * sizeof(ClipTriangleData),
                              "cudaMallocHost clip outputs");

  PackedClipJob *host_jobs = runtime.host_jobs.as<PackedClipJob>();
  size_t output_offset = 0;
  for (size_t local_index = 0; local_index < job_count; ++local_index) {
    const ClipBatchInput &input = inputs[begin + local_index];
    const DeviceMeshView view = device_mesh_view(*input.device_mesh);
    wait_for_device_mesh(*input.device_mesh, runtime.stream);
    host_jobs[local_index] = {
        reinterpret_cast<const double3 *>(view.vertices),
        reinterpret_cast<const int3 *>(view.triangles),
        make_double4(input.plane.a, input.plane.b, input.plane.c,
                     input.plane.d),
        static_cast<int>(output_offset),
        static_cast<int>(view.triangle_count)};
    output_offset += view.triangle_count;
  }

  check_cuda(cudaMemcpyAsync(runtime.jobs.as<PackedClipJob>(), host_jobs,
                             job_count * sizeof(PackedClipJob),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy clip jobs");
  prepare_clip_kernel<<<static_cast<unsigned int>(job_count), 256, 0,
                        runtime.stream>>>(
      runtime.jobs.as<PackedClipJob>(),
      runtime.outputs.as<ClipTriangleData>(), static_cast<int>(job_count));
  check_cuda(cudaGetLastError(), "launch clip preparation kernel");
  check_cuda(cudaMemcpyAsync(runtime.host_outputs.as<ClipTriangleData>(),
                             runtime.outputs.as<ClipTriangleData>(),
                             triangle_count * sizeof(ClipTriangleData),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy clip outputs");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "cudaStreamSynchronize clip batch");

  const ClipTriangleData *host_outputs =
      runtime.host_outputs.as<ClipTriangleData>();
  output_offset = 0;
  for (size_t index = begin; index < end; ++index) {
    const size_t count =
        device_mesh_view(*inputs[index].device_mesh).triangle_count;
    inputs[index].output->assign(host_outputs + output_offset,
                                 host_outputs + output_offset + count);
    output_offset += count;
  }
}

} // namespace

void prepare_clip_batch(const std::vector<ClipBatchInput> &inputs,
                        ClipBatchRuntime &runtime, size_t max_batch_size,
                        double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0)
    throw std::invalid_argument("Clip memory fraction must be in (0, 1]");
  validate_inputs(inputs);

  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo clip batch");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
    size_t end = begin;
    size_t triangle_count = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const size_t next_triangles =
          triangle_count +
          device_mesh_view(*inputs[end].device_mesh).triangle_count;
      const size_t growth =
          runtime.impl_->growth(end - begin + 1, next_triangles);
      if (end > begin && growth > budget)
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
