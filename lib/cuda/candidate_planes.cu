#include <algorithm>
#include <candidate_planes.hpp>
#include <cmath>
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
      check_cuda(cudaFree(data_), "cudaFree candidate buffer");
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
      check_cuda(cudaFreeHost(data_), "cudaFreeHost candidate buffer");
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

struct PackedCandidateJob {
  const double3 *vertices;
  const uint2 *edges;
  const unsigned int *sampled_edges;
  double4 *planes;
  int vertex_count;
  int edge_count;
  int sample_count;
  int max_planes;
  int job_index;
};

__device__ double add_rn(double first, double second) {
  return __dadd_rn(first, second);
}

__device__ double multiply_rn(double first, double second) {
  return __dmul_rn(first, second);
}

__device__ double plane_dot(const double4 &first, const double4 &second) {
  return add_rn(add_rn(multiply_rn(first.x, second.x),
                       multiply_rn(first.y, second.y)),
                multiply_rn(first.z, second.z));
}

__global__ void generate_candidates_kernel(
    const PackedCandidateJob *jobs, int *plane_counts, int *attempt_counts,
    int job_count, double normal_epsilon, double distance_epsilon) {
  const int job_index = static_cast<int>(blockIdx.x);
  if (job_index >= job_count)
    return;
  const PackedCandidateJob job = jobs[job_index];

  __shared__ int accepted_count;
  __shared__ int attempts;
  __shared__ int valid;
  __shared__ int too_similar;
  __shared__ double4 candidate;
  if (threadIdx.x == 0) {
    accepted_count = 0;
    attempts = 0;
  }
  __syncthreads();

  for (int sample_index = 0; sample_index < job.sample_count;
       ++sample_index) {
    if (accepted_count >= job.max_planes)
      break;

    if (threadIdx.x == 0) {
      attempts = sample_index + 1;
      valid = 0;
      const unsigned int sampled_edge = job.sampled_edges[sample_index];
      if (sampled_edge < static_cast<unsigned int>(job.edge_count)) {
        const uint2 edge = job.edges[sampled_edge];
        if (edge.x < static_cast<unsigned int>(job.vertex_count) &&
            edge.y < static_cast<unsigned int>(job.vertex_count)) {
          const double3 first = job.vertices[edge.x];
          const double3 second = job.vertices[edge.y];
          const double dx = add_rn(second.x, -first.x);
          const double dy = add_rn(second.y, -first.y);
          const double dz = add_rn(second.z, -first.z);
          const double squared_length =
              add_rn(add_rn(multiply_rn(dx, dx), multiply_rn(dy, dy)),
                     multiply_rn(dz, dz));
          const double length = sqrt(squared_length);
          if (length >= 1e-6) {
            const double nx = __ddiv_rn(dx, length);
            const double ny = __ddiv_rn(dy, length);
            const double nz = __ddiv_rn(dz, length);
            const double mx =
                multiply_rn(add_rn(first.x, second.x), 0.5);
            const double my =
                multiply_rn(add_rn(first.y, second.y), 0.5);
            const double mz =
                multiply_rn(add_rn(first.z, second.z), 0.5);
            const double distance =
                -add_rn(add_rn(multiply_rn(nx, mx), multiply_rn(ny, my)),
                        multiply_rn(nz, mz));
            candidate = make_double4(nx, ny, nz, distance);
            valid = 1;
          }
        }
      }
      too_similar = 0;
    }
    __syncthreads();

    if (valid) {
      for (int plane_index = static_cast<int>(threadIdx.x);
           plane_index < accepted_count; plane_index += blockDim.x) {
        const double4 existing = job.planes[plane_index];
        const double normal_dot = fabs(plane_dot(candidate, existing));
        if (normal_dot > normal_epsilon &&
            fabs(candidate.w - existing.w) < distance_epsilon) {
          atomicExch(&too_similar, 1);
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0 && valid && !too_similar) {
      job.planes[accepted_count] = candidate;
      ++accepted_count;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    plane_counts[job.job_index] = accepted_count;
    attempt_counts[job.job_index] = attempts;
  }
}

size_t growth_bytes(const DeviceBuffer &buffer, size_t requested) {
  return requested > buffer.capacity() ? requested - buffer.capacity() : 0;
}

} // namespace

struct CandidatePlaneRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer edges;
  DeviceBuffer sampled_edges;
  DeviceBuffer jobs;
  DeviceBuffer planes;
  DeviceBuffer plane_counts;
  DeviceBuffer attempt_counts;
  PinnedBuffer host_vertices;
  PinnedBuffer host_edges;
  PinnedBuffer host_sampled_edges;
  PinnedBuffer host_jobs;
  PinnedBuffer host_planes;
  PinnedBuffer host_plane_counts;
  PinnedBuffer host_attempt_counts;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags candidates");
    }
  }

  size_t growth(size_t vertex_count, size_t edge_count, size_t sample_count,
                size_t plane_count, size_t job_count) const {
    size_t result = 0;
    const auto include = [&](const DeviceBuffer &buffer, size_t count,
                             size_t element_size) {
      result = checked_add(
          result,
          growth_bytes(buffer,
                       checked_multiply(count, element_size,
                                        "Candidate allocation overflow")),
          "Candidate allocation total overflow");
    };
    include(vertices, vertex_count, sizeof(double3));
    include(edges, edge_count, sizeof(uint2));
    include(sampled_edges, sample_count, sizeof(unsigned int));
    include(jobs, job_count, sizeof(PackedCandidateJob));
    include(planes, plane_count, sizeof(double4));
    include(plane_counts, job_count, sizeof(int));
    include(attempt_counts, job_count, sizeof(int));
    return result;
  }
};

CandidatePlaneRuntime::CandidatePlaneRuntime()
    : impl_(std::make_unique<Impl>()) {}
CandidatePlaneRuntime::~CandidatePlaneRuntime() = default;
CandidatePlaneRuntime::CandidatePlaneRuntime(
    CandidatePlaneRuntime &&) noexcept = default;
CandidatePlaneRuntime &
CandidatePlaneRuntime::operator=(CandidatePlaneRuntime &&) noexcept = default;

namespace {

bool add_int_count(size_t current, size_t addition, size_t &result) {
  if (addition >
      static_cast<size_t>(std::numeric_limits<int>::max()) - current) {
    return false;
  }
  result = current + addition;
  return true;
}

void validate_inputs(const std::vector<CandidatePlaneInput> &inputs) {
  for (const CandidatePlaneInput &input : inputs) {
    if (!input.mesh || !input.planes || !input.attempts_used)
      throw std::invalid_argument("Candidate input contains a null pointer");
    if (input.sample_count && !input.sampled_edges)
      throw std::invalid_argument("Candidate samples are null");
    if (input.mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.mesh->intersecting_edges.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.sample_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.max_planes >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("Candidate input exceeds indexing limits");
    }
    for (size_t index = 0; index < input.sample_count; ++index) {
      if (input.sampled_edges[index] >=
          input.mesh->intersecting_edges.size()) {
        throw std::invalid_argument(
            "Candidate sample contains an invalid edge index");
      }
    }
    if (input.device_mesh) {
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      if (view.vertex_count != input.mesh->vertices.size() ||
          view.edge_count != input.mesh->intersecting_edges.size()) {
        throw std::invalid_argument(
            "Candidate device mesh counts do not match the input");
      }
    }
  }
}

struct OutputRange {
  CandidatePlaneInput input;
  size_t plane_offset;
  size_t job_index;
};

void run_wave(const std::vector<CandidatePlaneInput> &inputs, size_t begin,
              size_t end, CandidatePlaneRuntime::Impl &runtime) {
  size_t vertex_count = 0;
  size_t edge_count = 0;
  size_t sample_count = 0;
  size_t plane_count = 0;
  size_t job_count = end - begin;
  if (job_count >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Packed candidate job count exceeds limits");
  }
  for (size_t index = begin; index < end; ++index) {
    const CandidatePlaneInput &input = inputs[index];
    if (!input.device_mesh) {
      if (!add_int_count(vertex_count, input.mesh->vertices.size(),
                         vertex_count) ||
          !add_int_count(edge_count, input.mesh->intersecting_edges.size(),
                         edge_count)) {
        throw std::overflow_error(
            "Packed candidate mesh data exceeds indexing limits");
      }
    }
    if (!add_int_count(sample_count, input.sample_count, sample_count) ||
        !add_int_count(plane_count, input.max_planes, plane_count)) {
      throw std::overflow_error(
          "Packed candidate output exceeds indexing limits");
    }
  }

  runtime.ensure_stream();
  runtime.vertices.ensure(
      checked_multiply(vertex_count, sizeof(double3),
                       "Candidate vertex allocation overflow"),
      "cudaMalloc candidate vertices");
  runtime.edges.ensure(
      checked_multiply(edge_count, sizeof(uint2),
                       "Candidate edge allocation overflow"),
      "cudaMalloc candidate edges");
  runtime.sampled_edges.ensure(
      checked_multiply(sample_count, sizeof(unsigned int),
                       "Candidate sample allocation overflow"),
      "cudaMalloc candidate samples");
  runtime.jobs.ensure(
      checked_multiply(job_count, sizeof(PackedCandidateJob),
                       "Candidate job allocation overflow"),
      "cudaMalloc candidate jobs");
  runtime.planes.ensure(
      checked_multiply(plane_count, sizeof(double4),
                       "Candidate plane allocation overflow"),
      "cudaMalloc candidate planes");
  runtime.plane_counts.ensure(
      checked_multiply(job_count, sizeof(int),
                       "Candidate count allocation overflow"),
      "cudaMalloc candidate counts");
  runtime.attempt_counts.ensure(
      checked_multiply(job_count, sizeof(int),
                       "Candidate attempt allocation overflow"),
      "cudaMalloc candidate attempts");
  runtime.host_vertices.ensure(vertex_count * sizeof(double3),
                               "cudaMallocHost candidate vertices");
  runtime.host_edges.ensure(edge_count * sizeof(uint2),
                            "cudaMallocHost candidate edges");
  runtime.host_sampled_edges.ensure(
      sample_count * sizeof(unsigned int),
      "cudaMallocHost candidate samples");
  runtime.host_jobs.ensure(job_count * sizeof(PackedCandidateJob),
                           "cudaMallocHost candidate jobs");
  runtime.host_planes.ensure(plane_count * sizeof(double4),
                             "cudaMallocHost candidate planes");
  runtime.host_plane_counts.ensure(job_count * sizeof(int),
                                   "cudaMallocHost candidate counts");
  runtime.host_attempt_counts.ensure(job_count * sizeof(int),
                                     "cudaMallocHost candidate attempts");

  double3 *host_vertices = runtime.host_vertices.as<double3>();
  uint2 *host_edges = runtime.host_edges.as<uint2>();
  unsigned int *host_samples =
      runtime.host_sampled_edges.as<unsigned int>();
  PackedCandidateJob *host_jobs =
      runtime.host_jobs.as<PackedCandidateJob>();
  std::vector<OutputRange> outputs;
  outputs.reserve(job_count);

  size_t vertex_offset = 0;
  size_t edge_offset = 0;
  size_t sample_offset = 0;
  size_t plane_offset = 0;
  for (size_t index = begin; index < end; ++index) {
    const CandidatePlaneInput &input = inputs[index];
    const double3 *vertices = nullptr;
    const uint2 *edges = nullptr;
    if (input.device_mesh) {
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      wait_for_device_mesh(*input.device_mesh, runtime.stream);
      vertices = reinterpret_cast<const double3 *>(view.vertices);
      edges = reinterpret_cast<const uint2 *>(view.edges);
    } else {
      vertices = runtime.vertices.as<double3>() + vertex_offset;
      edges = runtime.edges.as<uint2>() + edge_offset;
      for (const Vec3D &vertex : input.mesh->vertices) {
        host_vertices[vertex_offset++] =
            make_double3(vertex[0], vertex[1], vertex[2]);
      }
      for (const auto &edge : input.mesh->intersecting_edges) {
        host_edges[edge_offset++] = make_uint2(edge.first, edge.second);
      }
    }
    if (input.sample_count) {
      std::copy(input.sampled_edges,
                input.sampled_edges + input.sample_count,
                host_samples + sample_offset);
    }
    const size_t job_index = index - begin;
    host_jobs[job_index] = {
        vertices,
        edges,
        runtime.sampled_edges.as<unsigned int>() + sample_offset,
        runtime.planes.as<double4>() + plane_offset,
        static_cast<int>(input.mesh->vertices.size()),
        static_cast<int>(input.mesh->intersecting_edges.size()),
        static_cast<int>(input.sample_count),
        static_cast<int>(input.max_planes),
        static_cast<int>(job_index)};
    outputs.push_back({input, plane_offset, job_index});
    sample_offset += input.sample_count;
    plane_offset += input.max_planes;
  }

  if (vertex_count) {
    check_cuda(cudaMemcpyAsync(runtime.vertices.as<double3>(), host_vertices,
                               vertex_count * sizeof(double3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy candidate vertices");
  }
  if (edge_count) {
    check_cuda(cudaMemcpyAsync(runtime.edges.as<uint2>(), host_edges,
                               edge_count * sizeof(uint2),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy candidate edges");
  }
  if (sample_count) {
    check_cuda(cudaMemcpyAsync(runtime.sampled_edges.as<unsigned int>(),
                               host_samples,
                               sample_count * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy candidate samples");
  }
  check_cuda(cudaMemcpyAsync(runtime.jobs.as<PackedCandidateJob>(), host_jobs,
                             job_count * sizeof(PackedCandidateJob),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy candidate jobs");

  constexpr int block_size = 256;
  const double normal_epsilon =
      std::cos(5.0 * 3.14159265358979323846 / 180.0);
  generate_candidates_kernel<<<static_cast<unsigned int>(job_count),
                               block_size, 0, runtime.stream>>>(
      runtime.jobs.as<PackedCandidateJob>(),
      runtime.plane_counts.as<int>(), runtime.attempt_counts.as<int>(),
      static_cast<int>(job_count), normal_epsilon, 1e-3);
  check_cuda(cudaGetLastError(), "launch candidate generation");
  check_cuda(cudaMemcpyAsync(runtime.host_planes.as<double4>(),
                             runtime.planes.as<double4>(),
                             plane_count * sizeof(double4),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy candidate planes");
  check_cuda(cudaMemcpyAsync(runtime.host_plane_counts.as<int>(),
                             runtime.plane_counts.as<int>(),
                             job_count * sizeof(int), cudaMemcpyDeviceToHost,
                             runtime.stream),
             "copy candidate counts");
  check_cuda(cudaMemcpyAsync(runtime.host_attempt_counts.as<int>(),
                             runtime.attempt_counts.as<int>(),
                             job_count * sizeof(int), cudaMemcpyDeviceToHost,
                             runtime.stream),
             "copy candidate attempts");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "cudaStreamSynchronize candidates");

  const double4 *host_planes = runtime.host_planes.as<double4>();
  const int *host_counts = runtime.host_plane_counts.as<int>();
  const int *host_attempts = runtime.host_attempt_counts.as<int>();
  for (const OutputRange &output : outputs) {
    const int count = host_counts[output.job_index];
    const int attempts = host_attempts[output.job_index];
    if (count < 0 ||
        static_cast<size_t>(count) > output.input.max_planes ||
        attempts < 0 ||
        static_cast<size_t>(attempts) > output.input.sample_count) {
      throw std::runtime_error(
          "Candidate kernel returned an invalid output count");
    }
    output.input.planes->clear();
    output.input.planes->reserve(static_cast<size_t>(count));
    for (int plane_index = 0; plane_index < count; ++plane_index) {
      const double4 plane =
          host_planes[output.plane_offset + plane_index];
      output.input.planes->emplace_back(plane.x, plane.y, plane.z, plane.w);
    }
    *output.input.attempts_used = static_cast<size_t>(attempts);
  }
}

} // namespace

void generate_candidate_planes_batch(
    const std::vector<CandidatePlaneInput> &inputs,
    CandidatePlaneRuntime &runtime, size_t max_batch_size,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Candidate memory fraction must be in (0, 1]");
  }
  validate_inputs(inputs);

  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo candidates");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
    size_t end = begin;
    size_t vertices = 0;
    size_t edges = 0;
    size_t samples = 0;
    size_t planes = 0;
    size_t jobs = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      size_t next_vertices = vertices;
      size_t next_edges = edges;
      size_t next_samples = samples;
      size_t next_planes = planes;
      if ((!inputs[end].device_mesh &&
           (!add_int_count(vertices, inputs[end].mesh->vertices.size(),
                           next_vertices) ||
            !add_int_count(edges,
                           inputs[end].mesh->intersecting_edges.size(),
                           next_edges))) ||
          !add_int_count(samples, inputs[end].sample_count, next_samples) ||
          !add_int_count(planes, inputs[end].max_planes, next_planes)) {
        break;
      }
      const size_t growth =
          runtime.impl_->growth(next_vertices, next_edges, next_samples,
                                next_planes, jobs + 1);
      if (end > begin && growth > budget)
        break;
      vertices = next_vertices;
      edges = next_edges;
      samples = next_samples;
      planes = next_planes;
      ++jobs;
      ++end;
    }
    if (end == begin)
      ++end;
    run_wave(inputs, begin, end, *runtime.impl_);
    begin = end;
  }
}

} // namespace neural_acd
