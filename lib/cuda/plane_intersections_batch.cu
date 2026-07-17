#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <device_mesh.hpp>
#include <limits>
#include <memory>
#include <plane_intersections.hpp>
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

struct PackedPlaneJob {
  const float3 *points;
  const uint2 *edges;
  int num_points;
  int num_edges;
};

__global__ void plane_score_packed_kernel(
    const float4 *planes, const PackedPlaneJob *jobs, const int *plane_jobs,
    float *scores, int num_planes, float eps = 1e-6f) {
  const int plane_idx = static_cast<int>(blockIdx.x);
  if (plane_idx >= num_planes)
    return;

  const PackedPlaneJob job = jobs[plane_jobs[plane_idx]];
  const float4 plane = planes[plane_idx];
  float thread_score = 0.0f;
  for (int edge_idx = threadIdx.x; edge_idx < job.num_edges;
       edge_idx += blockDim.x) {
    const uint2 edge = job.edges[edge_idx];
    if (edge.x >= static_cast<unsigned int>(job.num_points) ||
        edge.y >= static_cast<unsigned int>(job.num_points)) {
      continue;
    }

    const float3 p1 = job.points[edge.x];
    const float3 p2 = job.points[edge.y];
    const float value1 =
        plane.x * p1.x + plane.y * p1.y + plane.z * p1.z + plane.w;
    const float value2 =
        plane.x * p2.x + plane.y * p2.y + plane.z * p2.z + plane.w;
    const int side1 = (value1 > eps) - (value1 < -eps);
    const int side2 = (value2 > eps) - (value2 < -eps);
    if (side1 == 0 || side2 == 0 || side1 == side2)
      continue;

    const float dx = p2.x - p1.x;
    const float dy = p2.y - p1.y;
    const float dz = p2.z - p1.z;
    thread_score += sqrtf(dx * dx + dy * dy + dz * dz);
  }

  __shared__ float partial_scores[256];
  partial_scores[threadIdx.x] = thread_score;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial_scores[threadIdx.x] +=
          partial_scores[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scores[plane_idx] = partial_scores[0];
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (data_)
      cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void ensure(size_t bytes, const char *operation) {
    if (bytes <= capacity_)
      return;
    if (data_) {
      check_cuda(cudaFree(data_), "cudaFree pooled device buffer");
      data_ = nullptr;
      capacity_ = 0;
    }
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
  PinnedBuffer() = default;
  ~PinnedBuffer() {
    if (data_)
      cudaFreeHost(data_);
  }

  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  void ensure(size_t bytes, const char *operation) {
    if (bytes <= capacity_)
      return;
    if (data_) {
      check_cuda(cudaFreeHost(data_), "cudaFreeHost pooled buffer");
      data_ = nullptr;
      capacity_ = 0;
    }
    check_cuda(cudaMallocHost(&data_, bytes), operation);
    capacity_ = bytes;
  }

  template <typename T> T *as() const { return static_cast<T *>(data_); }

private:
  void *data_ = nullptr;
  size_t capacity_ = 0;
};

size_t growth_bytes(const DeviceBuffer &buffer, size_t requested) {
  return requested > buffer.capacity() ? requested - buffer.capacity() : 0;
}

bool is_active(const PlaneScoreInput &input) {
  return input.num_planes > 0 && input.num_edges > 0;
}

} // namespace

struct PlaneScoringRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer planes;
  DeviceBuffer points;
  DeviceBuffer edges;
  DeviceBuffer jobs;
  DeviceBuffer plane_jobs;
  DeviceBuffer scores;
  PinnedBuffer host_planes;
  PinnedBuffer host_points;
  PinnedBuffer host_edges;
  PinnedBuffer host_jobs;
  PinnedBuffer host_plane_jobs;
  PinnedBuffer host_scores;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags plane scoring");
    }
  }

  size_t additional_device_bytes(size_t num_planes, size_t num_points,
                                 size_t num_edges, size_t num_jobs) const {
    return growth_bytes(planes, sizeof(float4) * num_planes) +
           growth_bytes(points, sizeof(float3) * num_points) +
           growth_bytes(edges, sizeof(uint2) * num_edges) +
           growth_bytes(jobs, sizeof(PackedPlaneJob) * num_jobs) +
           growth_bytes(plane_jobs, sizeof(int) * num_planes) +
           growth_bytes(scores, sizeof(float) * num_planes);
  }
};

PlaneScoringRuntime::PlaneScoringRuntime() : impl_(std::make_unique<Impl>()) {}
PlaneScoringRuntime::~PlaneScoringRuntime() = default;
PlaneScoringRuntime::PlaneScoringRuntime(PlaneScoringRuntime &&) noexcept =
    default;
PlaneScoringRuntime &
PlaneScoringRuntime::operator=(PlaneScoringRuntime &&) noexcept = default;

namespace {

struct ScoreRange {
  float *destination;
  size_t packed_offset;
  size_t count;
};

void validate_inputs(const std::vector<PlaneScoreInput> &inputs) {
  for (const auto &input : inputs) {
    if (input.num_planes < 0 || input.num_points < 0 || input.num_edges < 0)
      throw std::invalid_argument("Plane scoring counts cannot be negative");
    if ((input.num_planes && (!input.planes || !input.scores)) ||
        (!input.device_mesh && input.num_points && !input.points) ||
        (!input.device_mesh && input.num_edges && !input.edges)) {
      throw std::invalid_argument("Plane scoring input contains a null buffer");
    }
    if (input.device_mesh) {
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      if (view.vertex_count != static_cast<size_t>(input.num_points) ||
          view.edge_count != static_cast<size_t>(input.num_edges)) {
        throw std::invalid_argument(
            "Plane scoring device mesh counts do not match the input");
      }
    }
  }
}

bool add_count(size_t current, int addition, size_t &result) {
  const size_t value = static_cast<size_t>(addition);
  if (value > static_cast<size_t>(std::numeric_limits<int>::max()) - current)
    return false;
  result = current + value;
  return true;
}

void run_wave(const std::vector<PlaneScoreInput> &inputs, size_t begin,
              size_t end, PlaneScoringRuntime::Impl &runtime) {
  size_t total_planes = 0;
  size_t total_points = 0;
  size_t total_edges = 0;
  size_t total_jobs = 0;
  for (size_t i = begin; i < end; ++i) {
    const PlaneScoreInput &input = inputs[i];
    if (!is_active(input))
      continue;
    size_t next = 0;
    if (!add_count(total_planes, input.num_planes, next))
      throw std::overflow_error("Packed plane count exceeds CUDA grid limit");
    total_planes = next;
    if (!input.device_mesh) {
      if (!add_count(total_points, input.num_points, next))
        throw std::overflow_error("Packed point count exceeds indexing limit");
      total_points = next;
      if (!add_count(total_edges, input.num_edges, next))
        throw std::overflow_error("Packed edge count exceeds indexing limit");
      total_edges = next;
    }
    ++total_jobs;
  }
  if (total_planes == 0)
    return;

  runtime.ensure_stream();
  runtime.host_planes.ensure(sizeof(float4) * total_planes,
                             "cudaMallocHost packed planes");
  runtime.host_points.ensure(sizeof(float3) * total_points,
                             "cudaMallocHost packed points");
  runtime.host_edges.ensure(sizeof(uint2) * total_edges,
                            "cudaMallocHost packed edges");
  runtime.host_jobs.ensure(sizeof(PackedPlaneJob) * total_jobs,
                           "cudaMallocHost packed jobs");
  runtime.host_plane_jobs.ensure(sizeof(int) * total_planes,
                                 "cudaMallocHost plane job map");
  runtime.host_scores.ensure(sizeof(float) * total_planes,
                             "cudaMallocHost packed scores");

  runtime.planes.ensure(sizeof(float4) * total_planes,
                        "cudaMalloc packed planes");
  runtime.points.ensure(sizeof(float3) * total_points,
                        "cudaMalloc packed points");
  runtime.edges.ensure(sizeof(uint2) * total_edges,
                       "cudaMalloc packed edges");
  runtime.jobs.ensure(sizeof(PackedPlaneJob) * total_jobs,
                      "cudaMalloc packed jobs");
  runtime.plane_jobs.ensure(sizeof(int) * total_planes,
                            "cudaMalloc plane job map");
  runtime.scores.ensure(sizeof(float) * total_planes,
                        "cudaMalloc packed scores");

  float *host_planes = runtime.host_planes.as<float>();
  float *host_points = runtime.host_points.as<float>();
  unsigned int *host_edges = runtime.host_edges.as<unsigned int>();
  PackedPlaneJob *host_jobs = runtime.host_jobs.as<PackedPlaneJob>();
  int *host_plane_jobs = runtime.host_plane_jobs.as<int>();
  std::vector<ScoreRange> score_ranges;
  score_ranges.reserve(total_jobs);

  size_t plane_offset = 0;
  size_t point_offset = 0;
  size_t edge_offset = 0;
  size_t job_index = 0;
  for (size_t i = begin; i < end; ++i) {
    const PlaneScoreInput &input = inputs[i];
    if (!is_active(input))
      continue;

    const float3 *job_points = nullptr;
    const uint2 *job_edges = nullptr;
    if (input.device_mesh) {
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      wait_for_device_mesh(*input.device_mesh, runtime.stream);
      job_points = reinterpret_cast<const float3 *>(view.float_vertices);
      job_edges = reinterpret_cast<const uint2 *>(view.edges);
    } else {
      job_points = runtime.points.as<float3>() + point_offset;
      job_edges = runtime.edges.as<uint2>() + edge_offset;
    }
    host_jobs[job_index] = {job_points, job_edges, input.num_points,
                            input.num_edges};
    std::memcpy(host_planes + plane_offset * 4, input.planes,
                sizeof(float4) * static_cast<size_t>(input.num_planes));
    if (!input.device_mesh && input.num_points) {
      std::memcpy(host_points + point_offset * 3, input.points,
                  sizeof(float3) * static_cast<size_t>(input.num_points));
    }
    if (!input.device_mesh) {
      std::memcpy(host_edges + edge_offset * 2, input.edges,
                  sizeof(uint2) * static_cast<size_t>(input.num_edges));
    }
    std::fill(host_plane_jobs + plane_offset,
              host_plane_jobs + plane_offset + input.num_planes,
              static_cast<int>(job_index));
    score_ranges.push_back(
        {input.scores, plane_offset, static_cast<size_t>(input.num_planes)});

    plane_offset += static_cast<size_t>(input.num_planes);
    if (!input.device_mesh) {
      point_offset += static_cast<size_t>(input.num_points);
      edge_offset += static_cast<size_t>(input.num_edges);
    }
    ++job_index;
  }

  check_cuda(cudaMemcpyAsync(runtime.planes.as<float4>(), host_planes,
                             sizeof(float4) * total_planes,
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy packed planes");
  if (total_points) {
    check_cuda(cudaMemcpyAsync(runtime.points.as<float3>(), host_points,
                               sizeof(float3) * total_points,
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy packed points");
  }
  check_cuda(cudaMemcpyAsync(runtime.edges.as<uint2>(), host_edges,
                             sizeof(uint2) * total_edges,
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy packed edges");
  check_cuda(cudaMemcpyAsync(runtime.jobs.as<PackedPlaneJob>(), host_jobs,
                             sizeof(PackedPlaneJob) * total_jobs,
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy packed jobs");
  check_cuda(cudaMemcpyAsync(runtime.plane_jobs.as<int>(), host_plane_jobs,
                             sizeof(int) * total_planes,
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy plane job map");

  constexpr int block_size = 256;
  plane_score_packed_kernel<<<static_cast<unsigned int>(total_planes),
                              block_size, 0, runtime.stream>>>(
      runtime.planes.as<float4>(), runtime.jobs.as<PackedPlaneJob>(),
      runtime.plane_jobs.as<int>(), runtime.scores.as<float>(),
      static_cast<int>(total_planes));
  check_cuda(cudaGetLastError(), "packed plane scoring kernel launch");
  check_cuda(cudaMemcpyAsync(runtime.host_scores.as<float>(),
                             runtime.scores.as<float>(),
                             sizeof(float) * total_planes,
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy packed plane scores");
  check_cuda(cudaStreamSynchronize(runtime.stream), "packed plane scoring");

  const float *host_scores = runtime.host_scores.as<float>();
  for (const ScoreRange &range : score_ranges) {
    std::memcpy(range.destination, host_scores + range.packed_offset,
                sizeof(float) * range.count);
  }
}

} // namespace

void classify_and_rate_planes_batch(
    const std::vector<PlaneScoreInput> &inputs, PlaneScoringRuntime &runtime,
    size_t max_batch_size, double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0)
    throw std::invalid_argument("batch_memory_fraction must be in (0, 1]");
  validate_inputs(inputs);

  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);

    size_t end = begin;
    size_t total_planes = 0;
    size_t total_points = 0;
    size_t total_edges = 0;
    size_t total_jobs = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;

      size_t next_planes = total_planes;
      size_t next_points = total_points;
      size_t next_edges = total_edges;
      size_t next_jobs = total_jobs;
      if (is_active(inputs[end])) {
        if (!add_count(total_planes, inputs[end].num_planes, next_planes)) {
          break;
        }
        if (!inputs[end].device_mesh &&
            (!add_count(total_points, inputs[end].num_points, next_points) ||
             !add_count(total_edges, inputs[end].num_edges, next_edges))) {
          break;
        }
        ++next_jobs;
      }
      const size_t growth = runtime.impl_->additional_device_bytes(
          next_planes, next_points, next_edges, next_jobs);
      if (end > begin && growth > budget)
        break;

      total_planes = next_planes;
      total_points = next_points;
      total_edges = next_edges;
      total_jobs = next_jobs;
      ++end;
    }
    if (end == begin)
      ++end;

    run_wave(inputs, begin, end, *runtime.impl_);
    begin = end;
  }
}

void classify_and_rate_planes_batch(
    const std::vector<PlaneScoreInput> &inputs, size_t max_batch_size,
    double memory_fraction) {
  PlaneScoringRuntime runtime;
  classify_and_rate_planes_batch(inputs, runtime, max_batch_size,
                                 memory_fraction);
}

} // namespace neural_acd
