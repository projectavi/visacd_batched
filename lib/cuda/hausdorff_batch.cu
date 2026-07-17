#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>
#include <device_mesh.hpp>
#include <hausdorff_batch.hpp>
#include <limits>
#include <memory>
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

struct PackedHausdorffQuery {
  double3 point;
  double nearest_sample_distance_squared;
  const double3 *target_vertices;
  const int3 *target_triangles;
  int candidate_offset;
  int candidate_count;
};

struct PackedHausdorffJob {
  int query_offset;
  int query_count;
};

__device__ double squared_length(double3 value) {
  return value.x * value.x + value.y * value.y + value.z * value.z;
}

__device__ double3 subtract(double3 first, double3 second) {
  return make_double3(first.x - second.x, first.y - second.y,
                      first.z - second.z);
}

__device__ double dot_product(double3 first, double3 second) {
  return first.x * second.x + first.y * second.y + first.z * second.z;
}

__device__ double point_distance(double3 point, double3 target) {
  return sqrt(squared_length(subtract(point, target)));
}

__device__ double point_segment_distance(double3 point, double3 first,
                                         double3 second) {
  const double3 point_from_second = subtract(point, second);
  const double3 first_from_second = subtract(first, second);
  const double segment_length = sqrt(squared_length(first_from_second));
  const double projection =
      dot_product(point_from_second, first_from_second) / segment_length;
  if (projection < 0.0 || projection > segment_length)
    return DBL_MAX;
  const double point_length = sqrt(squared_length(point_from_second));
  return sqrt(point_length * point_length - projection * projection);
}

__device__ bool point_in_triangle(double3 point, double3 first,
                                  double3 second, double3 third) {
  const double3 v0 = subtract(third, first);
  const double3 v1 = subtract(second, first);
  const double3 v2 = subtract(point, first);
  const double dot00 = dot_product(v0, v0);
  const double dot01 = dot_product(v0, v1);
  const double dot02 = dot_product(v0, v2);
  const double dot11 = dot_product(v1, v1);
  const double dot12 = dot_product(v1, v2);
  const double inverse = 1.0 / (dot00 * dot11 - dot01 * dot01);
  const double u = (dot11 * dot02 - dot01 * dot12) * inverse;
  const double v = (dot00 * dot12 - dot01 * dot02) * inverse;
  return u >= 0.0 && v >= 0.0 && u + v <= 1.0;
}

__device__ double point_triangle_distance(double3 point, double3 first,
                                          double3 second, double3 third) {
  const double ax = (second.y - first.y) * (third.z - first.z) -
                    (second.z - first.z) * (third.y - first.y);
  const double ay = (second.z - first.z) * (third.x - first.x) -
                    (second.x - first.x) * (third.z - first.z);
  const double az = (second.x - first.x) * (third.y - first.y) -
                    (second.y - first.y) * (third.x - first.x);
  const double normal_length = sqrt(ax * ax + ay * ay + az * az);
  const double a = ax / normal_length;
  const double b = ay / normal_length;
  const double c = az / normal_length;
  const double d = -(a * first.x + b * first.y + c * first.z);
  const double signed_distance =
      a * point.x + b * point.y + c * point.z + d;
  const double distance = fabs(signed_distance);
  double3 projected = point;
  if (signed_distance > 1e-8) {
    projected.x -= a * distance;
    projected.y -= b * distance;
    projected.z -= c * distance;
  } else if (signed_distance < -1e-8) {
    projected.x += a * distance;
    projected.y += b * distance;
    projected.z += c * distance;
  }

  if (point_in_triangle(projected, first, second, third))
    return distance;

  double result = point_segment_distance(point, first, second);
  result = fmin(result, point_segment_distance(point, second, third));
  result = fmin(result, point_segment_distance(point, third, first));
  result = fmin(result, point_distance(point, first));
  result = fmin(result, point_distance(point, second));
  result = fmin(result, point_distance(point, third));
  return result;
}

__global__ void evaluate_queries_kernel(
    const PackedHausdorffQuery *queries, const int *candidate_triangles,
    double *query_distances, int query_count) {
  const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= query_count)
    return;

  const PackedHausdorffQuery query = queries[query_index];
  double closest = DBL_MAX;
  for (int candidate_index = 0; candidate_index < query.candidate_count;
       ++candidate_index) {
    const int triangle_index =
        candidate_triangles[query.candidate_offset + candidate_index];
    const int3 triangle = query.target_triangles[triangle_index];
    const double distance = point_triangle_distance(
        query.point, query.target_vertices[triangle.x],
        query.target_vertices[triangle.y],
        query.target_vertices[triangle.z]);
    closest = fmin(closest, distance);
    if (closest < 1e-14)
      break;
  }
  if (closest > 10.0)
    closest = sqrt(query.nearest_sample_distance_squared);
  query_distances[query_index] =
      closest < DBL_MAX && !isnan(closest) ? closest : 0.0;
}

__global__ void reduce_jobs_kernel(const double *query_distances,
                                   const PackedHausdorffJob *jobs,
                                   double *results, int job_count) {
  const int job_index = blockIdx.x;
  if (job_index >= job_count)
    return;
  const PackedHausdorffJob job = jobs[job_index];
  double local_maximum = 0.0;
  for (int query_index = threadIdx.x; query_index < job.query_count;
       query_index += blockDim.x) {
    local_maximum =
        fmax(local_maximum, query_distances[job.query_offset + query_index]);
  }

  __shared__ double partial[kThreads];
  partial[threadIdx.x] = local_maximum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] =
          fmax(partial[threadIdx.x], partial[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[job_index] = partial[0];
}

class DeviceBuffer {
public:
  ~DeviceBuffer() {
    if (data_)
      cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer() = default;

  void ensure(size_t bytes, const char *operation) {
    if (bytes <= capacity_)
      return;
    if (data_)
      check_cuda(cudaFree(data_), "cudaFree Hausdorff device buffer");
    data_ = nullptr;
    capacity_ = 0;
    check_cuda(cudaMalloc(&data_, bytes), operation);
    capacity_ = bytes;
  }

  template <typename T> T *as() const { return static_cast<T *>(data_); }

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

  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  PinnedBuffer() = default;

  void ensure(size_t bytes, const char *operation) {
    if (bytes <= capacity_)
      return;
    if (data_)
      check_cuda(cudaFreeHost(data_), "cudaFreeHost Hausdorff buffer");
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

size_t checked_add(size_t first, size_t second, const char *message) {
  if (second > std::numeric_limits<size_t>::max() - first)
    throw std::overflow_error(message);
  return first + second;
}

size_t job_device_bytes(const PreparedHausdorffJob &job) {
  size_t bytes = sizeof(PackedHausdorffJob) + sizeof(double);
  for (const PreparedHausdorffDirection &direction : job.directions) {
    if (!direction.target)
      continue;
    if (!direction.target_device) {
      bytes = checked_add(bytes,
                          direction.target->vertices.size() * sizeof(double3),
                          "Hausdorff vertex bytes overflow");
      bytes = checked_add(
          bytes, direction.target->triangles.size() * sizeof(int3),
          "Hausdorff triangle bytes overflow");
    }
    bytes = checked_add(
        bytes,
        direction.queries.size() *
            (sizeof(PackedHausdorffQuery) + sizeof(double) +
             kHausdorffCandidateCount * sizeof(int)),
        "Hausdorff query bytes overflow");
  }
  return bytes;
}

} // namespace

struct HausdorffRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer triangles;
  DeviceBuffer queries;
  DeviceBuffer candidates;
  DeviceBuffer query_distances;
  DeviceBuffer jobs;
  DeviceBuffer results;
  PinnedBuffer host_vertices;
  PinnedBuffer host_triangles;
  PinnedBuffer host_queries;
  PinnedBuffer host_candidates;
  PinnedBuffer host_jobs;
  PinnedBuffer host_results;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags Hausdorff");
    }
  }
};

HausdorffRuntime::HausdorffRuntime() : impl_(std::make_unique<Impl>()) {}
HausdorffRuntime::~HausdorffRuntime() = default;
HausdorffRuntime::HausdorffRuntime(HausdorffRuntime &&) noexcept = default;
HausdorffRuntime &
HausdorffRuntime::operator=(HausdorffRuntime &&) noexcept = default;

namespace {

void validate_direction(const PreparedHausdorffDirection &direction) {
  const size_t query_count = direction.queries.size();
  if (!direction.target ||
      direction.candidate_triangles.size() != query_count ||
      direction.candidate_counts.size() != query_count ||
      direction.nearest_sample_distance_squared.size() != query_count) {
    throw std::invalid_argument("Malformed prepared Hausdorff direction");
  }
  if (direction.target_device) {
    const DeviceMeshView view = device_mesh_view(*direction.target_device);
    if (view.vertex_count != direction.target->vertices.size() ||
        view.triangle_count != direction.target->triangles.size()) {
      throw std::invalid_argument(
          "Hausdorff device mesh does not match its target");
    }
  }
}

void run_wave(const std::vector<PreparedHausdorffJob *> &jobs, size_t begin,
              size_t end, HausdorffRuntime::Impl &runtime) {
  std::vector<PreparedHausdorffJob *> active;
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  size_t query_count = 0;
  for (size_t index = begin; index < end; ++index) {
    PreparedHausdorffJob &job = *jobs[index];
    if (!job.valid) {
      job.result = INF;
      continue;
    }
    for (const PreparedHausdorffDirection &direction : job.directions) {
      validate_direction(direction);
      if (!direction.target_device) {
        vertex_count = checked_add(vertex_count,
                                   direction.target->vertices.size(),
                                   "Packed Hausdorff vertices overflow");
        triangle_count = checked_add(
            triangle_count, direction.target->triangles.size(),
            "Packed Hausdorff triangles overflow");
      }
      query_count = checked_add(query_count, direction.queries.size(),
                                "Packed Hausdorff queries overflow");
    }
    active.push_back(&job);
  }
  if (active.empty())
    return;
  if (vertex_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      triangle_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      query_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Packed Hausdorff input exceeds indexing limits");
  }

  if (query_count >
      std::numeric_limits<size_t>::max() / kHausdorffCandidateCount) {
    throw std::overflow_error("Hausdorff candidate count overflow");
  }
  const size_t candidate_count = query_count * kHausdorffCandidateCount;
  if (candidate_count >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Packed Hausdorff candidates exceed indexing limits");
  }
  runtime.ensure_stream();
  runtime.host_vertices.ensure(vertex_count * sizeof(double3),
                               "cudaMallocHost Hausdorff vertices");
  runtime.host_triangles.ensure(triangle_count * sizeof(int3),
                                "cudaMallocHost Hausdorff triangles");
  runtime.host_queries.ensure(query_count * sizeof(PackedHausdorffQuery),
                              "cudaMallocHost Hausdorff queries");
  runtime.host_candidates.ensure(candidate_count * sizeof(int),
                                 "cudaMallocHost Hausdorff candidates");
  runtime.host_jobs.ensure(active.size() * sizeof(PackedHausdorffJob),
                           "cudaMallocHost Hausdorff jobs");
  runtime.host_results.ensure(active.size() * sizeof(double),
                              "cudaMallocHost Hausdorff results");
  runtime.vertices.ensure(vertex_count * sizeof(double3),
                          "cudaMalloc Hausdorff vertices");
  runtime.triangles.ensure(triangle_count * sizeof(int3),
                           "cudaMalloc Hausdorff triangles");
  runtime.queries.ensure(query_count * sizeof(PackedHausdorffQuery),
                         "cudaMalloc Hausdorff queries");
  runtime.candidates.ensure(candidate_count * sizeof(int),
                            "cudaMalloc Hausdorff candidates");
  runtime.query_distances.ensure(query_count * sizeof(double),
                                 "cudaMalloc Hausdorff distances");
  runtime.jobs.ensure(active.size() * sizeof(PackedHausdorffJob),
                      "cudaMalloc Hausdorff jobs");
  runtime.results.ensure(active.size() * sizeof(double),
                         "cudaMalloc Hausdorff results");

  double3 *host_vertices = runtime.host_vertices.as<double3>();
  int3 *host_triangles = runtime.host_triangles.as<int3>();
  PackedHausdorffQuery *host_queries =
      runtime.host_queries.as<PackedHausdorffQuery>();
  int *host_candidates = runtime.host_candidates.as<int>();
  PackedHausdorffJob *host_jobs =
      runtime.host_jobs.as<PackedHausdorffJob>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  size_t query_offset = 0;
  size_t candidate_offset = 0;
  for (size_t job_index = 0; job_index < active.size(); ++job_index) {
    const size_t job_query_offset = query_offset;
    for (const PreparedHausdorffDirection &direction :
         active[job_index]->directions) {
      const Mesh &target = *direction.target;
      const double3 *target_vertices = nullptr;
      const int3 *target_triangles = nullptr;
      if (direction.target_device) {
        const DeviceMeshView view = device_mesh_view(*direction.target_device);
        wait_for_device_mesh(*direction.target_device, runtime.stream);
        target_vertices =
            reinterpret_cast<const double3 *>(view.vertices);
        target_triangles =
            reinterpret_cast<const int3 *>(view.triangles);
      } else {
        const size_t direction_vertex_offset = vertex_offset;
        for (const Vec3D &vertex : target.vertices) {
          host_vertices[vertex_offset++] =
              make_double3(vertex[0], vertex[1], vertex[2]);
        }
        const size_t direction_triangle_offset = triangle_offset;
        for (const auto &triangle : target.triangles) {
          host_triangles[triangle_offset++] =
              make_int3(triangle[0], triangle[1], triangle[2]);
        }
        target_vertices = runtime.vertices.as<double3>() +
                          direction_vertex_offset;
        target_triangles = runtime.triangles.as<int3>() +
                           direction_triangle_offset;
      }

      for (size_t local_query = 0; local_query < direction.queries.size();
           ++local_query) {
        const Vec3D &point = direction.queries[local_query];
        const int count = direction.candidate_counts[local_query];
        if (count > static_cast<int>(kHausdorffCandidateCount))
          throw std::invalid_argument("Invalid Hausdorff candidate count");
        host_queries[query_offset++] =
            {make_double3(point[0], point[1], point[2]),
             direction.nearest_sample_distance_squared[local_query],
             target_vertices, target_triangles,
             static_cast<int>(candidate_offset), count};
        for (size_t candidate = 0; candidate < kHausdorffCandidateCount;
             ++candidate) {
          const int triangle =
              direction.candidate_triangles[local_query][candidate];
          if (candidate < static_cast<size_t>(count) &&
              (triangle < 0 ||
               triangle >= static_cast<int>(target.triangles.size()))) {
            throw std::out_of_range("Hausdorff candidate triangle is invalid");
          }
          host_candidates[candidate_offset++] = triangle;
        }
      }
    }
    host_jobs[job_index] =
        {static_cast<int>(job_query_offset),
         static_cast<int>(query_offset - job_query_offset)};
  }

  if (vertex_count) {
    check_cuda(cudaMemcpyAsync(runtime.vertices.as<double3>(), host_vertices,
                               vertex_count * sizeof(double3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "cudaMemcpyAsync Hausdorff vertices");
  }
  if (triangle_count) {
    check_cuda(cudaMemcpyAsync(runtime.triangles.as<int3>(), host_triangles,
                               triangle_count * sizeof(int3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "cudaMemcpyAsync Hausdorff triangles");
  }
  check_cuda(cudaMemcpyAsync(runtime.queries.as<PackedHausdorffQuery>(),
                             host_queries,
                             query_count * sizeof(PackedHausdorffQuery),
                             cudaMemcpyHostToDevice, runtime.stream),
             "cudaMemcpyAsync Hausdorff queries");
  check_cuda(cudaMemcpyAsync(runtime.candidates.as<int>(), host_candidates,
                             candidate_count * sizeof(int),
                             cudaMemcpyHostToDevice, runtime.stream),
             "cudaMemcpyAsync Hausdorff candidates");
  check_cuda(cudaMemcpyAsync(runtime.jobs.as<PackedHausdorffJob>(), host_jobs,
                             active.size() * sizeof(PackedHausdorffJob),
                             cudaMemcpyHostToDevice, runtime.stream),
             "cudaMemcpyAsync Hausdorff jobs");

  const int blocks =
      (static_cast<int>(query_count) + kThreads - 1) / kThreads;
  evaluate_queries_kernel<<<blocks, kThreads, 0, runtime.stream>>>(
      runtime.queries.as<PackedHausdorffQuery>(), runtime.candidates.as<int>(),
      runtime.query_distances.as<double>(), static_cast<int>(query_count));
  check_cuda(cudaGetLastError(), "launch Hausdorff query kernel");
  reduce_jobs_kernel<<<static_cast<int>(active.size()), kThreads, 0,
                       runtime.stream>>>(
      runtime.query_distances.as<double>(),
      runtime.jobs.as<PackedHausdorffJob>(), runtime.results.as<double>(),
      static_cast<int>(active.size()));
  check_cuda(cudaGetLastError(), "launch Hausdorff reduction kernel");

  double *host_results = runtime.host_results.as<double>();
  check_cuda(cudaMemcpyAsync(host_results, runtime.results.as<double>(),
                             active.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "cudaMemcpyAsync Hausdorff results");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "cudaStreamSynchronize Hausdorff");
  for (size_t index = 0; index < active.size(); ++index)
    active[index]->result = host_results[index];
}

} // namespace

void evaluate_hausdorff_batch(
    const std::vector<PreparedHausdorffJob *> &jobs,
    HausdorffRuntime &runtime, size_t max_batch_size, double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0)
    throw std::invalid_argument("Hausdorff memory fraction must be in (0, 1]");
  for (PreparedHausdorffJob *job : jobs) {
    if (!job)
      throw std::invalid_argument("Hausdorff job cannot be null");
  }

  size_t begin = 0;
  while (begin < jobs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo Hausdorff");
    const size_t budget =
        std::max<size_t>(1, static_cast<size_t>(free_bytes * memory_fraction));
    const size_t hard_end =
        max_batch_size == 0
            ? jobs.size()
            : std::min(jobs.size(), begin + max_batch_size);
    size_t end = begin;
    size_t bytes = 0;
    while (end < hard_end) {
      const size_t next = job_device_bytes(*jobs[end]);
      if (end > begin && next > budget - std::min(bytes, budget))
        break;
      bytes = checked_add(bytes, next, "Hausdorff wave bytes overflow");
      ++end;
    }
    if (end == begin)
      ++end;
    run_wave(jobs, begin, end, *runtime.impl_);
    begin = end;
  }
}

} // namespace neural_acd
