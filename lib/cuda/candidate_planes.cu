#include <algorithm>
#include <candidate_planes.hpp>
#include <cmath>
#include <config.hpp>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_buffer.hpp>
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

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

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

struct ClipSideData {
  short sides[3];
};

struct ClipIntersectionRecord {
  int triangle_index;
  unsigned short intersection_mask;
  unsigned short padding;
  double intersections[9];
};

struct PackedSplitJob {
  const double3 *vertices;
  const float3 *float_vertices;
  const uint2 *edges;
  const int3 *triangles;
  const double4 *candidates;
  const double4 *flat_planes;
  float *scores;
  ClipSideData *clip_sides;
  ClipIntersectionRecord *clip_intersections;
  unsigned char *clip_flags;
  int clip_offset;
  int vertex_count;
  int edge_count;
  int triangle_count;
  int max_candidates;
  int flat_plane_count;
  float flat_surface_weight;
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

  __shared__ int attempts;
  __shared__ int valid;
  __shared__ int too_similar;
  __shared__ double4 candidate;
  int accepted_count = 0;
  if (threadIdx.x == 0) {
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

    const bool accepted = valid && !too_similar;
    if (threadIdx.x == 0 && accepted)
      job.planes[accepted_count] = candidate;
    if (accepted)
      ++accepted_count;
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    plane_counts[job.job_index] = accepted_count;
    attempt_counts[job.job_index] = attempts;
  }
}

__global__ void score_split_planes_kernel(const PackedSplitJob *jobs,
                                          const int *candidate_counts,
                                          int job_count,
                                          int max_plane_count,
                                          float eps = 1e-6f) {
  const int plane_index = static_cast<int>(blockIdx.x);
  const int job_index = static_cast<int>(blockIdx.y);
  if (job_index >= job_count || plane_index >= max_plane_count)
    return;

  const PackedSplitJob job = jobs[job_index];
  const int candidate_count = candidate_counts[job.job_index];
  const int plane_count = candidate_count + job.flat_plane_count;
  if (plane_index >= plane_count)
    return;

  const double4 source =
      plane_index < candidate_count
          ? job.candidates[plane_index]
          : job.flat_planes[plane_index - candidate_count];
  const float4 plane =
      make_float4(static_cast<float>(source.x),
                  static_cast<float>(source.y),
                  static_cast<float>(source.z),
                  static_cast<float>(source.w));
  float thread_score = 0.0f;
  for (int edge_index = static_cast<int>(threadIdx.x);
       edge_index < job.edge_count; edge_index += blockDim.x) {
    const uint2 edge = job.edges[edge_index];
    if (edge.x >= static_cast<unsigned int>(job.vertex_count) ||
        edge.y >= static_cast<unsigned int>(job.vertex_count)) {
      continue;
    }
    const float3 first = job.float_vertices[edge.x];
    const float3 second = job.float_vertices[edge.y];
    const float first_value =
        plane.x * first.x + plane.y * first.y + plane.z * first.z + plane.w;
    const float second_value = plane.x * second.x + plane.y * second.y +
                               plane.z * second.z + plane.w;
    const int first_side = (first_value > eps) - (first_value < -eps);
    const int second_side = (second_value > eps) - (second_value < -eps);
    if (first_side == 0 || second_side == 0 || first_side == second_side)
      continue;

    const float dx = second.x - first.x;
    const float dy = second.y - first.y;
    const float dz = second.z - first.z;
    thread_score += sqrtf(dx * dx + dy * dy + dz * dz);
  }

  __shared__ float partial_scores[256];
  partial_scores[threadIdx.x] = thread_score;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset)
      partial_scores[threadIdx.x] += partial_scores[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float score = partial_scores[0];
    if (plane_index >= candidate_count)
      score *= job.flat_surface_weight;
    job.scores[plane_index] = score;
  }
}

__global__ void select_split_planes_kernel(const PackedSplitJob *jobs,
                                           const int *candidate_counts,
                                           double4 *selected_planes,
                                           int job_count) {
  const int job_index = static_cast<int>(blockIdx.x);
  if (job_index >= job_count || threadIdx.x != 0)
    return;
  const PackedSplitJob job = jobs[job_index];
  const int candidate_count = candidate_counts[job.job_index];
  const int plane_count = candidate_count + job.flat_plane_count;
  if (plane_count == 0) {
    selected_planes[job_index] = make_double4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  int selected_index = 0;
  float selected_score = job.scores[0];
  for (int plane_index = 1; plane_index < plane_count; ++plane_index) {
    const float score = job.scores[plane_index];
    if (selected_score < score) {
      selected_score = score;
      selected_index = plane_index;
    }
  }
  selected_planes[job_index] =
      selected_index < candidate_count
          ? job.candidates[selected_index]
          : job.flat_planes[selected_index - candidate_count];
}

__device__ double subtract_rn(double first, double second) {
  return __dadd_rn(first, -second);
}

__device__ double split_plane_value(double3 point, double4 plane) {
  double value = multiply_rn(point.x, plane.x);
  value = add_rn(value, multiply_rn(point.y, plane.y));
  value = add_rn(value, multiply_rn(point.z, plane.z));
  return add_rn(value, plane.w);
}

__device__ short split_point_side(double3 point, double4 plane) {
  const double value = split_plane_value(point, plane);
  if (value > 1e-6)
    return 1;
  if (value < -1e-6)
    return -1;
  return 0;
}

__device__ short split_coplanar_side(double3 first, double3 second,
                                     double3 third, double4 plane) {
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

__device__ double split_segment_denominator(double3 first, double3 second,
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

__device__ bool split_segment_intersection(double3 first, double3 second,
                                           double4 plane,
                                           double3 &intersection) {
  const double denominator = split_segment_denominator(first, second, plane);
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

__device__ void split_store_intersection(ClipIntersectionRecord &output,
                                         int edge, double3 point) {
  output.intersections[edge * 3] = point.x;
  output.intersections[edge * 3 + 1] = point.y;
  output.intersections[edge * 3 + 2] = point.z;
}

__global__ void prepare_selected_clips_kernel(
    const PackedSplitJob *jobs, const int *candidate_counts,
    const double4 *selected_planes, int job_count) {
  const int job_index = static_cast<int>(blockIdx.x);
  if (job_index >= job_count)
    return;
  const PackedSplitJob job = jobs[job_index];
  if (candidate_counts[job.job_index] + job.flat_plane_count == 0)
    return;
  const double4 plane = selected_planes[job_index];
  for (int local_triangle = static_cast<int>(threadIdx.x);
       local_triangle < job.triangle_count;
       local_triangle += static_cast<int>(blockDim.x)) {
    const int3 triangle = job.triangles[local_triangle];
    const double3 points[3] = {job.vertices[triangle.x],
                               job.vertices[triangle.y],
                               job.vertices[triangle.z]};
    ClipSideData sides;
    sides.sides[0] = split_point_side(points[0], plane);
    sides.sides[1] = split_point_side(points[1], plane);
    sides.sides[2] = split_point_side(points[2], plane);
    if (sides.sides[0] == 0 && sides.sides[1] == 0 &&
        sides.sides[2] == 0) {
      const short side =
          split_coplanar_side(points[0], points[1], points[2], plane);
      sides.sides[0] = side;
      sides.sides[1] = side;
      sides.sides[2] = side;
    }
    job.clip_sides[local_triangle] = sides;
    const short sum = sides.sides[0] + sides.sides[1] + sides.sides[2];
    const bool positive_side =
        sum == 3 || sum == 2 ||
        (sum == 1 &&
         ((sides.sides[0] == 1 && sides.sides[1] == 0 &&
           sides.sides[2] == 0) ||
          (sides.sides[0] == 0 && sides.sides[1] == 1 &&
           sides.sides[2] == 0) ||
          (sides.sides[0] == 0 && sides.sides[1] == 0 &&
           sides.sides[2] == 1)));
    const bool negative_side =
        sum == -3 || sum == -2 ||
        (sum == -1 &&
         ((sides.sides[0] == -1 && sides.sides[1] == 0 &&
           sides.sides[2] == 0) ||
          (sides.sides[0] == 0 && sides.sides[1] == -1 &&
           sides.sides[2] == 0) ||
          (sides.sides[0] == 0 && sides.sides[1] == 0 &&
           sides.sides[2] == -1)));
    if (positive_side || negative_side) {
      job.clip_flags[local_triangle] = 0;
      continue;
    }

    ClipIntersectionRecord output{};
    output.triangle_index = job.clip_offset + local_triangle;
    double3 intersection;
    if (split_segment_intersection(points[0], points[1], plane,
                                   intersection)) {
      output.intersection_mask |= 1u;
      split_store_intersection(output, 0, intersection);
    }
    if (split_segment_intersection(points[1], points[2], plane,
                                   intersection)) {
      output.intersection_mask |= 2u;
      split_store_intersection(output, 1, intersection);
    }
    if (split_segment_intersection(points[2], points[0], plane,
                                   intersection)) {
      output.intersection_mask |= 4u;
      split_store_intersection(output, 2, intersection);
    }
    job.clip_intersections[local_triangle] = output;
    job.clip_flags[local_triangle] = 1;
  }
}

size_t growth_bytes(const DeviceBuffer &buffer, size_t requested) {
  return requested > buffer.capacity() ? requested - buffer.capacity() : 0;
}

} // namespace

struct CandidatePlaneRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer float_vertices;
  DeviceBuffer edges;
  DeviceBuffer triangles;
  DeviceBuffer sampled_edges;
  DeviceBuffer jobs;
  DeviceBuffer split_jobs;
  DeviceBuffer planes;
  DeviceBuffer flat_planes;
  DeviceBuffer scores;
  DeviceBuffer selected_planes;
  DeviceBuffer clip_sides;
  DeviceBuffer clip_outputs;
  DeviceBuffer compact_clip_outputs;
  DeviceBuffer clip_flags;
  DeviceBuffer compact_clip_count;
  DeviceBuffer clip_select_temp;
  DeviceBuffer plane_counts;
  DeviceBuffer attempt_counts;
  PinnedBuffer host_vertices;
  PinnedBuffer host_float_vertices;
  PinnedBuffer host_edges;
  PinnedBuffer host_triangles;
  PinnedBuffer host_sampled_edges;
  PinnedBuffer host_jobs;
  PinnedBuffer host_split_jobs;
  PinnedBuffer host_planes;
  PinnedBuffer host_flat_planes;
  PinnedBuffer host_selected_planes;
  PinnedBuffer host_clip_sides;
  PinnedBuffer host_compact_clip_outputs;
  PinnedBuffer host_compact_clip_count;
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
    DeviceBuffer::set_allocation_stream(stream);
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

  size_t split_growth(size_t vertex_count, size_t edge_count,
                      size_t packed_triangle_count,
                      size_t clip_triangle_count, size_t sample_count,
                      size_t candidate_count, size_t flat_plane_count,
                      size_t score_count, size_t job_count) const {
    size_t result = growth(vertex_count, edge_count, sample_count,
                           candidate_count, job_count);
    const auto include = [&](const DeviceBuffer &buffer, size_t count,
                             size_t element_size) {
      result = checked_add(
          result,
          growth_bytes(buffer,
                       checked_multiply(count, element_size,
                                        "Fused split allocation overflow")),
          "Fused split allocation total overflow");
    };
    include(float_vertices, vertex_count, sizeof(float3));
    include(triangles, packed_triangle_count, sizeof(int3));
    include(split_jobs, job_count, sizeof(PackedSplitJob));
    include(flat_planes, flat_plane_count, sizeof(double4));
    include(scores, score_count, sizeof(float));
    include(selected_planes, job_count, sizeof(double4));
    include(clip_sides, clip_triangle_count, sizeof(ClipSideData));
    include(clip_outputs, clip_triangle_count,
            sizeof(ClipIntersectionRecord));
    include(compact_clip_outputs, clip_triangle_count,
            sizeof(ClipIntersectionRecord));
    include(clip_flags, clip_triangle_count, sizeof(unsigned char));
    include(compact_clip_count, 1, sizeof(int));
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
             "cudaStreamSynchronize candidate counts");

  double4 *host_planes = runtime.host_planes.as<double4>();
  const int *host_counts = runtime.host_plane_counts.as<int>();
  const int *host_attempts = runtime.host_attempt_counts.as<int>();
  bool copied_planes = false;
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
    if (count > 0) {
      check_cuda(
          cudaMemcpyAsync(
              host_planes + output.plane_offset,
              runtime.planes.as<double4>() + output.plane_offset,
              checked_multiply(static_cast<size_t>(count), sizeof(double4),
                               "Candidate plane copy overflow"),
              cudaMemcpyDeviceToHost, runtime.stream),
          "copy accepted candidate planes");
      copied_planes = true;
    }
  }
  if (copied_planes) {
    check_cuda(cudaStreamSynchronize(runtime.stream),
               "cudaStreamSynchronize candidate planes");
  }

  for (const OutputRange &output : outputs) {
    const int count = host_counts[output.job_index];
    const int attempts = host_attempts[output.job_index];
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

namespace {

void validate_split_inputs(const std::vector<SplitPlaneInput> &inputs) {
  for (const SplitPlaneInput &input : inputs) {
    if (!input.mesh || !input.selected_plane ||
        !input.has_selected_plane || !input.prepared_clip ||
        !input.attempts_used) {
      throw std::invalid_argument("Fused split input contains a null pointer");
    }
    if (input.sample_count && !input.sampled_edges)
      throw std::invalid_argument("Fused split samples are null");
    const size_t flat_plane_count =
        input.flat_planes ? input.flat_planes->size() : 0;
    if (input.mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.mesh->intersecting_edges.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.mesh->triangles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.sample_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input.max_candidates >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        flat_plane_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        flat_plane_count >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                input.max_candidates) {
      throw std::overflow_error("Fused split input exceeds indexing limits");
    }
    for (size_t index = 0; index < input.sample_count; ++index) {
      if (input.sampled_edges[index] >=
          input.mesh->intersecting_edges.size()) {
        throw std::invalid_argument(
            "Fused split sample contains an invalid edge index");
      }
    }
    if (input.device_mesh) {
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      if (view.vertex_count != input.mesh->vertices.size() ||
          view.edge_count != input.mesh->intersecting_edges.size() ||
          view.triangle_count != input.mesh->triangles.size()) {
        throw std::invalid_argument(
            "Fused split device mesh counts do not match the input");
      }
    }
  }
}

struct SplitOutputRange {
  SplitPlaneInput input;
  size_t clip_offset;
  size_t job_index;
};

void run_split_wave(const std::vector<SplitPlaneInput> &inputs, size_t begin,
                    size_t end, CandidatePlaneRuntime::Impl &runtime) {
  const size_t job_count = end - begin;
  if (job_count >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Fused split job count exceeds limits");
  }

  size_t packed_vertex_count = 0;
  size_t packed_edge_count = 0;
  size_t packed_triangle_count = 0;
  size_t clip_triangle_count = 0;
  size_t sample_count = 0;
  size_t candidate_capacity = 0;
  size_t flat_plane_count = 0;
  size_t score_capacity = 0;
  size_t max_plane_count = 0;
  for (size_t index = begin; index < end; ++index) {
    const SplitPlaneInput &input = inputs[index];
    const size_t input_flat_count =
        input.flat_planes ? input.flat_planes->size() : 0;
    if (!input.device_mesh) {
      packed_vertex_count = checked_add(
          packed_vertex_count, input.mesh->vertices.size(),
          "Fused split packed vertex count overflow");
      packed_edge_count = checked_add(
          packed_edge_count, input.mesh->intersecting_edges.size(),
          "Fused split packed edge count overflow");
      packed_triangle_count = checked_add(
          packed_triangle_count, input.mesh->triangles.size(),
          "Fused split packed triangle count overflow");
    }
    clip_triangle_count = checked_add(
        clip_triangle_count, input.mesh->triangles.size(),
        "Fused split clip triangle count overflow");
    sample_count = checked_add(sample_count, input.sample_count,
                               "Fused split sample count overflow");
    candidate_capacity =
        checked_add(candidate_capacity, input.max_candidates,
                    "Fused split candidate count overflow");
    flat_plane_count =
        checked_add(flat_plane_count, input_flat_count,
                    "Fused split flat-plane count overflow");
    const size_t input_plane_capacity =
        checked_add(input.max_candidates, input_flat_count,
                    "Fused split score count overflow");
    score_capacity = checked_add(score_capacity, input_plane_capacity,
                                 "Fused split score count overflow");
    max_plane_count = std::max(max_plane_count, input_plane_capacity);
  }
  if (clip_triangle_count >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Fused split packed clip count exceeds limits");
  }

  runtime.ensure_stream();
  runtime.vertices.ensure(
      checked_multiply(packed_vertex_count, sizeof(double3),
                       "Fused split vertex allocation overflow"),
      "cudaMalloc fused split vertices");
  runtime.float_vertices.ensure(
      checked_multiply(packed_vertex_count, sizeof(float3),
                       "Fused split float vertex allocation overflow"),
      "cudaMalloc fused split float vertices");
  runtime.edges.ensure(
      checked_multiply(packed_edge_count, sizeof(uint2),
                       "Fused split edge allocation overflow"),
      "cudaMalloc fused split edges");
  runtime.triangles.ensure(
      checked_multiply(packed_triangle_count, sizeof(int3),
                       "Fused split triangle allocation overflow"),
      "cudaMalloc fused split triangles");
  runtime.sampled_edges.ensure(
      checked_multiply(sample_count, sizeof(unsigned int),
                       "Fused split sample allocation overflow"),
      "cudaMalloc fused split samples");
  runtime.jobs.ensure(
      checked_multiply(job_count, sizeof(PackedCandidateJob),
                       "Fused split candidate job allocation overflow"),
      "cudaMalloc fused split candidate jobs");
  runtime.split_jobs.ensure(
      checked_multiply(job_count, sizeof(PackedSplitJob),
                       "Fused split job allocation overflow"),
      "cudaMalloc fused split jobs");
  runtime.planes.ensure(
      checked_multiply(candidate_capacity, sizeof(double4),
                       "Fused split candidate allocation overflow"),
      "cudaMalloc fused split candidates");
  runtime.flat_planes.ensure(
      checked_multiply(flat_plane_count, sizeof(double4),
                       "Fused split flat-plane allocation overflow"),
      "cudaMalloc fused split flat planes");
  runtime.scores.ensure(
      checked_multiply(score_capacity, sizeof(float),
                       "Fused split score allocation overflow"),
      "cudaMalloc fused split scores");
  runtime.selected_planes.ensure(
      checked_multiply(job_count, sizeof(double4),
                       "Fused split selection allocation overflow"),
      "cudaMalloc fused split selected planes");
  runtime.clip_sides.ensure(
      checked_multiply(clip_triangle_count, sizeof(ClipSideData),
                       "Fused split clip side allocation overflow"),
      "cudaMalloc fused split clip sides");
  runtime.clip_outputs.ensure(
      checked_multiply(clip_triangle_count,
                       sizeof(ClipIntersectionRecord),
                       "Fused split clip allocation overflow"),
      "cudaMalloc fused split clip intersections");
  runtime.compact_clip_outputs.ensure(
      checked_multiply(clip_triangle_count,
                       sizeof(ClipIntersectionRecord),
                       "Fused split compact clip allocation overflow"),
      "cudaMalloc fused split compact clip intersections");
  runtime.clip_flags.ensure(
      checked_multiply(clip_triangle_count, sizeof(unsigned char),
                       "Fused split clip flag allocation overflow"),
      "cudaMalloc fused split clip flags");
  runtime.compact_clip_count.ensure(
      sizeof(int), "cudaMalloc fused split compact clip count");
  runtime.plane_counts.ensure(job_count * sizeof(int),
                              "cudaMalloc fused split candidate counts");
  runtime.attempt_counts.ensure(job_count * sizeof(int),
                                "cudaMalloc fused split attempt counts");

  runtime.host_vertices.ensure(packed_vertex_count * sizeof(double3),
                               "cudaMallocHost fused split vertices");
  runtime.host_float_vertices.ensure(
      packed_vertex_count * sizeof(float3),
      "cudaMallocHost fused split float vertices");
  runtime.host_edges.ensure(packed_edge_count * sizeof(uint2),
                            "cudaMallocHost fused split edges");
  runtime.host_triangles.ensure(packed_triangle_count * sizeof(int3),
                                "cudaMallocHost fused split triangles");
  runtime.host_sampled_edges.ensure(
      sample_count * sizeof(unsigned int),
      "cudaMallocHost fused split samples");
  runtime.host_jobs.ensure(job_count * sizeof(PackedCandidateJob),
                           "cudaMallocHost fused split candidate jobs");
  runtime.host_split_jobs.ensure(job_count * sizeof(PackedSplitJob),
                                 "cudaMallocHost fused split jobs");
  runtime.host_flat_planes.ensure(flat_plane_count * sizeof(double4),
                                  "cudaMallocHost fused split flat planes");
  runtime.host_selected_planes.ensure(
      job_count * sizeof(double4),
      "cudaMallocHost fused split selected planes");
  runtime.host_clip_sides.ensure(
      clip_triangle_count * sizeof(ClipSideData),
      "cudaMallocHost fused split clip sides");
  runtime.host_compact_clip_outputs.ensure(
      clip_triangle_count * sizeof(ClipIntersectionRecord),
      "cudaMallocHost fused split compact clip intersections");
  runtime.host_compact_clip_count.ensure(
      sizeof(int), "cudaMallocHost fused split compact clip count");
  runtime.host_plane_counts.ensure(
      job_count * sizeof(int),
      "cudaMallocHost fused split candidate counts");
  runtime.host_attempt_counts.ensure(
      job_count * sizeof(int),
      "cudaMallocHost fused split attempt counts");

  double3 *host_vertices = runtime.host_vertices.as<double3>();
  float3 *host_float_vertices =
      runtime.host_float_vertices.as<float3>();
  uint2 *host_edges = runtime.host_edges.as<uint2>();
  int3 *host_triangles = runtime.host_triangles.as<int3>();
  unsigned int *host_samples =
      runtime.host_sampled_edges.as<unsigned int>();
  PackedCandidateJob *host_candidate_jobs =
      runtime.host_jobs.as<PackedCandidateJob>();
  PackedSplitJob *host_split_jobs =
      runtime.host_split_jobs.as<PackedSplitJob>();
  double4 *host_flat_planes = runtime.host_flat_planes.as<double4>();

  std::vector<SplitOutputRange> outputs;
  outputs.reserve(job_count);
  size_t vertex_offset = 0;
  size_t edge_offset = 0;
  size_t triangle_offset = 0;
  size_t clip_offset = 0;
  size_t sample_offset = 0;
  size_t candidate_offset = 0;
  size_t flat_offset = 0;
  size_t score_offset = 0;
  for (size_t input_index = begin; input_index < end; ++input_index) {
    const SplitPlaneInput &input = inputs[input_index];
    const size_t local_job = input_index - begin;
    const size_t input_flat_count =
        input.flat_planes ? input.flat_planes->size() : 0;

    const double3 *vertices = nullptr;
    const float3 *float_vertices = nullptr;
    const uint2 *edges = nullptr;
    const int3 *triangles = nullptr;
    if (input.device_mesh) {
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      wait_for_device_mesh(*input.device_mesh, runtime.stream);
      vertices = reinterpret_cast<const double3 *>(view.vertices);
      float_vertices =
          reinterpret_cast<const float3 *>(view.float_vertices);
      edges = reinterpret_cast<const uint2 *>(view.edges);
      triangles = reinterpret_cast<const int3 *>(view.triangles);
    } else {
      vertices = runtime.vertices.as<double3>() + vertex_offset;
      float_vertices =
          runtime.float_vertices.as<float3>() + vertex_offset;
      edges = runtime.edges.as<uint2>() + edge_offset;
      triangles = runtime.triangles.as<int3>() + triangle_offset;
      for (size_t index = 0; index < input.mesh->vertices.size(); ++index) {
        const Vec3D &vertex = input.mesh->vertices[index];
        host_vertices[vertex_offset + index] =
            make_double3(vertex[0], vertex[1], vertex[2]);
        host_float_vertices[vertex_offset + index] =
            make_float3(static_cast<float>(vertex[0]),
                        static_cast<float>(vertex[1]),
                        static_cast<float>(vertex[2]));
      }
      for (size_t index = 0;
           index < input.mesh->intersecting_edges.size(); ++index) {
        const auto &edge = input.mesh->intersecting_edges[index];
        host_edges[edge_offset + index] =
            make_uint2(edge.first, edge.second);
      }
      for (size_t index = 0; index < input.mesh->triangles.size(); ++index) {
        const auto &triangle = input.mesh->triangles[index];
        host_triangles[triangle_offset + index] =
            make_int3(triangle[0], triangle[1], triangle[2]);
      }
    }

    if (input.sample_count) {
      std::copy(input.sampled_edges,
                input.sampled_edges + input.sample_count,
                host_samples + sample_offset);
    }
    if (input_flat_count) {
      for (size_t index = 0; index < input_flat_count; ++index) {
        const Plane &plane = (*input.flat_planes)[index];
        host_flat_planes[flat_offset + index] =
            make_double4(plane.a, plane.b, plane.c, plane.d);
      }
    }

    const double4 *flat_planes =
        input_flat_count ? runtime.flat_planes.as<double4>() + flat_offset
                         : nullptr;
    host_candidate_jobs[local_job] = {
        vertices,
        edges,
        input.sample_count
            ? runtime.sampled_edges.as<unsigned int>() + sample_offset
            : nullptr,
        runtime.planes.as<double4>() + candidate_offset,
        static_cast<int>(input.mesh->vertices.size()),
        static_cast<int>(input.mesh->intersecting_edges.size()),
        static_cast<int>(input.sample_count),
        static_cast<int>(input.max_candidates),
        static_cast<int>(local_job)};
    host_split_jobs[local_job] = {
        vertices,
        float_vertices,
        edges,
        triangles,
        runtime.planes.as<double4>() + candidate_offset,
        flat_planes,
        runtime.scores.as<float>() + score_offset,
        runtime.clip_sides.as<ClipSideData>() + clip_offset,
        runtime.clip_outputs.as<ClipIntersectionRecord>() + clip_offset,
        runtime.clip_flags.as<unsigned char>() + clip_offset,
        static_cast<int>(clip_offset),
        static_cast<int>(input.mesh->vertices.size()),
        static_cast<int>(input.mesh->intersecting_edges.size()),
        static_cast<int>(input.mesh->triangles.size()),
        static_cast<int>(input.max_candidates),
        static_cast<int>(input_flat_count),
        input.flat_surface_weight,
        static_cast<int>(local_job)};
    outputs.push_back({input, clip_offset, local_job});

    if (!input.device_mesh) {
      vertex_offset += input.mesh->vertices.size();
      edge_offset += input.mesh->intersecting_edges.size();
      triangle_offset += input.mesh->triangles.size();
    }
    clip_offset += input.mesh->triangles.size();
    sample_offset += input.sample_count;
    candidate_offset += input.max_candidates;
    flat_offset += input_flat_count;
    score_offset += input.max_candidates + input_flat_count;
  }

  if (packed_vertex_count) {
    check_cuda(cudaMemcpyAsync(runtime.vertices.as<double3>(), host_vertices,
                               packed_vertex_count * sizeof(double3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy fused split vertices");
    check_cuda(
        cudaMemcpyAsync(runtime.float_vertices.as<float3>(),
                        host_float_vertices,
                        packed_vertex_count * sizeof(float3),
                        cudaMemcpyHostToDevice, runtime.stream),
        "copy fused split float vertices");
  }
  if (packed_edge_count) {
    check_cuda(cudaMemcpyAsync(runtime.edges.as<uint2>(), host_edges,
                               packed_edge_count * sizeof(uint2),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy fused split edges");
  }
  if (packed_triangle_count) {
    check_cuda(cudaMemcpyAsync(runtime.triangles.as<int3>(), host_triangles,
                               packed_triangle_count * sizeof(int3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy fused split triangles");
  }
  if (sample_count) {
    check_cuda(cudaMemcpyAsync(runtime.sampled_edges.as<unsigned int>(),
                               host_samples,
                               sample_count * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy fused split samples");
  }
  if (flat_plane_count) {
    check_cuda(cudaMemcpyAsync(runtime.flat_planes.as<double4>(),
                               host_flat_planes,
                               flat_plane_count * sizeof(double4),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy fused split flat planes");
  }
  check_cuda(cudaMemcpyAsync(runtime.jobs.as<PackedCandidateJob>(),
                             host_candidate_jobs,
                             job_count * sizeof(PackedCandidateJob),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy fused split candidate jobs");
  check_cuda(cudaMemcpyAsync(runtime.split_jobs.as<PackedSplitJob>(),
                             host_split_jobs,
                             job_count * sizeof(PackedSplitJob),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy fused split jobs");
  if (clip_triangle_count) {
    check_cuda(cudaMemsetAsync(runtime.clip_flags.as<unsigned char>(), 0,
                               clip_triangle_count * sizeof(unsigned char),
                               runtime.stream),
               "clear fused split clip flags");
  }

  constexpr int block_size = 256;
  const double normal_epsilon =
      std::cos(5.0 * 3.14159265358979323846 / 180.0);
  generate_candidates_kernel<<<static_cast<unsigned int>(job_count),
                               block_size, 0, runtime.stream>>>(
      runtime.jobs.as<PackedCandidateJob>(),
      runtime.plane_counts.as<int>(), runtime.attempt_counts.as<int>(),
      static_cast<int>(job_count), normal_epsilon, 1e-3);
  check_cuda(cudaGetLastError(), "launch fused candidate generation");

  if (max_plane_count) {
    const dim3 grid(static_cast<unsigned int>(max_plane_count),
                    static_cast<unsigned int>(job_count));
    score_split_planes_kernel<<<grid, block_size, 0, runtime.stream>>>(
        runtime.split_jobs.as<PackedSplitJob>(),
        runtime.plane_counts.as<int>(), static_cast<int>(job_count),
        static_cast<int>(max_plane_count));
    check_cuda(cudaGetLastError(), "launch fused plane scoring");
  }
  select_split_planes_kernel<<<static_cast<unsigned int>(job_count), 1, 0,
                               runtime.stream>>>(
      runtime.split_jobs.as<PackedSplitJob>(),
      runtime.plane_counts.as<int>(),
      runtime.selected_planes.as<double4>(), static_cast<int>(job_count));
  check_cuda(cudaGetLastError(), "launch fused plane selection");
  prepare_selected_clips_kernel<<<static_cast<unsigned int>(job_count),
                                  block_size, 0, runtime.stream>>>(
      runtime.split_jobs.as<PackedSplitJob>(),
      runtime.plane_counts.as<int>(),
      runtime.selected_planes.as<double4>(), static_cast<int>(job_count));
  check_cuda(cudaGetLastError(), "launch fused clip preparation");

  if (clip_triangle_count) {
    size_t select_temp_bytes = 0;
    check_cuda(cub::DeviceSelect::Flagged(
                   nullptr, select_temp_bytes,
                   runtime.clip_outputs.as<ClipIntersectionRecord>(),
                   runtime.clip_flags.as<unsigned char>(),
                   runtime.compact_clip_outputs
                       .as<ClipIntersectionRecord>(),
                   runtime.compact_clip_count.as<int>(),
                   static_cast<int>(clip_triangle_count), runtime.stream),
               "query compact clip selection storage");
    runtime.clip_select_temp.ensure(
        select_temp_bytes, "cudaMalloc compact clip selection storage");
    check_cuda(cub::DeviceSelect::Flagged(
                   runtime.clip_select_temp.as<void>(), select_temp_bytes,
                   runtime.clip_outputs.as<ClipIntersectionRecord>(),
                   runtime.clip_flags.as<unsigned char>(),
                   runtime.compact_clip_outputs
                       .as<ClipIntersectionRecord>(),
                   runtime.compact_clip_count.as<int>(),
                   static_cast<int>(clip_triangle_count), runtime.stream),
               "compact clip intersections");
  }

  check_cuda(cudaMemcpyAsync(runtime.host_plane_counts.as<int>(),
                             runtime.plane_counts.as<int>(),
                             job_count * sizeof(int), cudaMemcpyDeviceToHost,
                             runtime.stream),
             "copy fused split candidate counts");
  check_cuda(cudaMemcpyAsync(runtime.host_attempt_counts.as<int>(),
                             runtime.attempt_counts.as<int>(),
                             job_count * sizeof(int), cudaMemcpyDeviceToHost,
                             runtime.stream),
             "copy fused split attempt counts");
  check_cuda(cudaMemcpyAsync(runtime.host_selected_planes.as<double4>(),
                             runtime.selected_planes.as<double4>(),
                             job_count * sizeof(double4),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy fused split selected planes");
  if (clip_triangle_count) {
    check_cuda(cudaMemcpyAsync(runtime.host_clip_sides.as<ClipSideData>(),
                               runtime.clip_sides.as<ClipSideData>(),
                               clip_triangle_count * sizeof(ClipSideData),
                               cudaMemcpyDeviceToHost, runtime.stream),
               "copy fused split clip sides");
    check_cuda(cudaMemcpyAsync(runtime.host_compact_clip_count.as<int>(),
                               runtime.compact_clip_count.as<int>(),
                               sizeof(int), cudaMemcpyDeviceToHost,
                               runtime.stream),
               "copy fused split compact clip count");
  }
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "synchronize fused split compact count");

  const int compact_clip_count =
      clip_triangle_count ? *runtime.host_compact_clip_count.as<int>() : 0;
  if (compact_clip_count < 0 ||
      static_cast<size_t>(compact_clip_count) > clip_triangle_count) {
    throw std::runtime_error(
        "Fused split kernel returned an invalid compact clip count");
  }
  if (config.batch_logging &&
      std::getenv("VISACD_CLIP_COMPACTION_DIAGNOSTICS")) {
    const size_t previous_bytes =
        clip_triangle_count * sizeof(ClipTriangleData);
    const size_t compact_bytes =
        clip_triangle_count * sizeof(ClipSideData) +
        static_cast<size_t>(compact_clip_count) *
            sizeof(ClipIntersectionRecord);
    std::fprintf(stderr,
                 "[visacd compact clip] triangles=%zu crossings=%d "
                 "d2h_bytes=%zu previous_bytes=%zu\n",
                 clip_triangle_count, compact_clip_count, compact_bytes,
                 previous_bytes);
  }
  if (compact_clip_count) {
    check_cuda(cudaMemcpyAsync(
                   runtime.host_compact_clip_outputs
                       .as<ClipIntersectionRecord>(),
                   runtime.compact_clip_outputs
                       .as<ClipIntersectionRecord>(),
                   static_cast<size_t>(compact_clip_count) *
                       sizeof(ClipIntersectionRecord),
                   cudaMemcpyDeviceToHost, runtime.stream),
               "copy fused split compact clip intersections");
    check_cuda(cudaStreamSynchronize(runtime.stream),
               "synchronize fused split compact intersections");
  }

  const int *host_counts = runtime.host_plane_counts.as<int>();
  const int *host_attempts = runtime.host_attempt_counts.as<int>();
  const double4 *host_selected =
      runtime.host_selected_planes.as<double4>();
  const ClipSideData *host_clip_sides =
      runtime.host_clip_sides.as<ClipSideData>();
  for (const SplitOutputRange &output : outputs) {
    const int candidate_count = host_counts[output.job_index];
    const int attempts = host_attempts[output.job_index];
    if (candidate_count < 0 ||
        static_cast<size_t>(candidate_count) >
            output.input.max_candidates ||
        attempts < 0 ||
        static_cast<size_t>(attempts) > output.input.sample_count) {
      throw std::runtime_error(
          "Fused candidate kernel returned an invalid output count");
    }
    const size_t input_flat_count =
        output.input.flat_planes ? output.input.flat_planes->size() : 0;
    const bool has_plane = candidate_count > 0 || input_flat_count > 0;
    *output.input.has_selected_plane = has_plane;
    *output.input.attempts_used = static_cast<size_t>(attempts);
    output.input.prepared_clip->clear();
    if (!has_plane)
      continue;

    const double4 selected = host_selected[output.job_index];
    *output.input.selected_plane =
        Plane(selected.x, selected.y, selected.z, selected.w);
    const size_t triangle_count = output.input.mesh->triangles.size();
    output.input.prepared_clip->resize(triangle_count);
    for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
      ClipTriangleData &clip = (*output.input.prepared_clip)[triangle];
      const ClipSideData &sides =
          host_clip_sides[output.clip_offset + triangle];
      clip.sides[0] = sides.sides[0];
      clip.sides[1] = sides.sides[1];
      clip.sides[2] = sides.sides[2];
    }
  }

  const ClipIntersectionRecord *host_intersections =
      runtime.host_compact_clip_outputs.as<ClipIntersectionRecord>();
  size_t output_index = 0;
  for (int compact = 0; compact < compact_clip_count; ++compact) {
    const ClipIntersectionRecord &record = host_intersections[compact];
    if (record.triangle_index < 0)
      throw std::runtime_error("Compact clip record has a negative index");
    const size_t triangle_index =
        static_cast<size_t>(record.triangle_index);
    while (output_index < outputs.size() &&
           triangle_index >=
               outputs[output_index].clip_offset +
                   outputs[output_index].input.mesh->triangles.size()) {
      ++output_index;
    }
    if (output_index >= outputs.size() ||
        triangle_index < outputs[output_index].clip_offset ||
        outputs[output_index].input.prepared_clip->empty()) {
      throw std::runtime_error("Compact clip record has an invalid owner");
    }
    ClipTriangleData &clip =
        (*outputs[output_index].input.prepared_clip)
            [triangle_index - outputs[output_index].clip_offset];
    clip.intersection_mask = record.intersection_mask;
    std::copy(std::begin(record.intersections),
              std::end(record.intersections),
              std::begin(clip.intersections));
  }
}

} // namespace

void generate_score_select_clip_batch(
    const std::vector<SplitPlaneInput> &inputs,
    CandidatePlaneRuntime &runtime, size_t max_batch_size,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Fused split memory fraction must be in (0, 1]");
  }
  validate_split_inputs(inputs);

  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo fused split");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);

    size_t end = begin;
    size_t vertices = 0;
    size_t edges = 0;
    size_t packed_triangles = 0;
    size_t clip_triangles = 0;
    size_t samples = 0;
    size_t candidates = 0;
    size_t flat_planes = 0;
    size_t scores = 0;
    size_t jobs = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const SplitPlaneInput &input = inputs[end];
      const size_t input_flat_count =
          input.flat_planes ? input.flat_planes->size() : 0;
      size_t next_vertices = vertices;
      size_t next_edges = edges;
      size_t next_packed_triangles = packed_triangles;
      size_t next_clip_triangles = clip_triangles;
      size_t next_samples = samples;
      size_t next_candidates = candidates;
      size_t next_flat_planes = flat_planes;
      size_t next_scores = scores;
      try {
        if (!input.device_mesh) {
          next_vertices = checked_add(
              vertices, input.mesh->vertices.size(),
              "Fused split wave vertex count overflow");
          next_edges = checked_add(
              edges, input.mesh->intersecting_edges.size(),
              "Fused split wave edge count overflow");
          next_packed_triangles = checked_add(
              packed_triangles, input.mesh->triangles.size(),
              "Fused split wave triangle count overflow");
        }
        next_clip_triangles = checked_add(
            clip_triangles, input.mesh->triangles.size(),
            "Fused split wave clip count overflow");
        if (next_clip_triangles >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
          break;
        }
        next_samples = checked_add(samples, input.sample_count,
                                   "Fused split wave sample count overflow");
        next_candidates =
            checked_add(candidates, input.max_candidates,
                        "Fused split wave candidate count overflow");
        next_flat_planes =
            checked_add(flat_planes, input_flat_count,
                        "Fused split wave flat-plane count overflow");
        next_scores = checked_add(
            scores, input.max_candidates + input_flat_count,
            "Fused split wave score count overflow");
      } catch (const std::overflow_error &) {
        break;
      }
      const size_t growth = runtime.impl_->split_growth(
          next_vertices, next_edges, next_packed_triangles,
          next_clip_triangles, next_samples, next_candidates,
          next_flat_planes, next_scores, jobs + 1);
      if (end > begin && growth > budget)
        break;
      vertices = next_vertices;
      edges = next_edges;
      packed_triangles = next_packed_triangles;
      clip_triangles = next_clip_triangles;
      samples = next_samples;
      candidates = next_candidates;
      flat_planes = next_flat_planes;
      scores = next_scores;
      ++jobs;
      ++end;
    }
    if (end == begin)
      ++end;
    run_split_wave(inputs, begin, end, *runtime.impl_);
    begin = end;
  }
}

} // namespace neural_acd
