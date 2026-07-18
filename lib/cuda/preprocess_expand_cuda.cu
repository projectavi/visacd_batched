#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <preprocess_expand_cuda.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace neural_acd {
namespace {

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

struct PackedNarrowbandCandidate {
  int x;
  int y;
  int z;
  int manhattan_limit;
  size_t fragment_offset;
  size_t fragment_count;
  int triangle_offset;
  double voxel_size;
};

__device__ double3 subtract(double3 first, double3 second) {
  return make_double3(first.x - second.x, first.y - second.y,
                      first.z - second.z);
}

__device__ double3 add_scaled(double3 point, double3 direction,
                              double scale) {
  return make_double3(point.x + direction.x * scale,
                      point.y + direction.y * scale,
                      point.z + direction.z * scale);
}

__device__ double dot_product(double3 first, double3 second) {
  return first.x * second.x + first.y * second.y +
         first.z * second.z;
}

__device__ double3 closest_triangle(double3 a, double3 b, double3 c,
                                    double3 point) {
  const double3 ab = subtract(b, a);
  const double3 ac = subtract(c, a);
  const double3 ap = subtract(point, a);
  const double d1 = dot_product(ab, ap);
  const double d2 = dot_product(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0)
    return a;

  const double3 bp = subtract(point, b);
  const double d3 = dot_product(ab, bp);
  const double d4 = dot_product(ac, bp);
  if (d3 >= 0.0 && d4 <= d3)
    return b;

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double parameter = d1 / (d1 - d3);
    return add_scaled(a, ab, parameter);
  }

  const double3 cp = subtract(point, c);
  const double d5 = dot_product(ab, cp);
  const double d6 = dot_product(ac, cp);
  if (d6 >= 0.0 && d5 <= d6)
    return c;

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double parameter = d2 / (d2 - d6);
    return add_scaled(a, ac, parameter);
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double parameter =
        (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return add_scaled(b, subtract(c, b), parameter);
  }

  const double inverse = 1.0 / (va + vb + vc);
  const double b_weight = vb * inverse;
  const double c_weight = vc * inverse;
  const double3 ab_term = make_double3(ab.x * b_weight,
                                       ab.y * b_weight,
                                       ab.z * b_weight);
  return add_scaled(make_double3(a.x + ab_term.x, a.y + ab_term.y,
                                 a.z + ab_term.z),
                    ac, c_weight);
}

__global__ void evaluate_narrowband_kernel(
    const float3 *vertices, const int3 *triangles,
    const NarrowbandFragment *fragments,
    const PackedNarrowbandCandidate *candidates, size_t candidate_count,
    NarrowbandDistance *distances) {
  const size_t candidate_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate_index >= candidate_count)
    return;

  const PackedNarrowbandCandidate candidate = candidates[candidate_index];
  const double3 point = make_double3(candidate.x, candidate.y,
                                     candidate.z);
  double closest_distance = DBL_MAX;
  int closest_triangle_index = 0;
  int last_triangle_index = -1;
  const size_t end = candidate.fragment_offset + candidate.fragment_count;
  for (size_t fragment_index = candidate.fragment_offset;
       fragment_index < end; ++fragment_index) {
    const NarrowbandFragment fragment = fragments[fragment_index];
    if (last_triangle_index == fragment.triangle_index)
      continue;
    const int manhattan = abs(fragment.x - candidate.x) +
                          abs(fragment.y - candidate.y) +
                          abs(fragment.z - candidate.z);
    if (manhattan > candidate.manhattan_limit)
      continue;
    last_triangle_index = fragment.triangle_index;

    const int3 triangle = triangles[fragment.triangle_index];
    const float3 af = vertices[triangle.x];
    const float3 bf = vertices[triangle.y];
    const float3 cf = vertices[triangle.z];
    const double3 a = make_double3(af.x, af.y, af.z);
    const double3 b = make_double3(bf.x, bf.y, bf.z);
    const double3 c = make_double3(cf.x, cf.y, cf.z);
    const double3 closest = closest_triangle(a, c, b, point);
    const double3 delta = subtract(point, closest);
    const double distance = dot_product(delta, delta);
    if (distance < closest_distance) {
      closest_distance = distance;
      closest_triangle_index = fragment.triangle_index;
    }
  }
  distances[candidate_index] =
      {sqrt(closest_distance) * candidate.voxel_size,
       closest_triangle_index - candidate.triangle_offset};
}

struct Runtime {
  std::mutex mutex;
  cudaStream_t stream = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer triangles;
  DeviceBuffer fragments;
  DeviceBuffer candidates;
  DeviceBuffer distances;
  PinnedBuffer host_vertices;
  PinnedBuffer host_triangles;
  PinnedBuffer host_fragments;
  PinnedBuffer host_candidates;
  PinnedBuffer host_distances;

  ~Runtime() {
    if (stream)
      cudaStreamDestroy(stream);
  }

  void ensure_stream() {
    if (!stream) {
      cuda_memory::check(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create narrowband CUDA stream");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }
};

Runtime &runtime() {
  static Runtime instance;
  return instance;
}

} // namespace

void evaluate_narrowband_distances_cuda(
    const Mesh &mesh, double scale, double voxel_size,
    const std::vector<NarrowbandFragment> &fragments,
    const std::vector<NarrowbandCandidate> &candidates,
    std::vector<NarrowbandDistance> &distances) {
  std::vector<NarrowbandEvaluationInput> inputs{
      {&mesh, scale, voxel_size, &fragments, &candidates, &distances}};
  evaluate_narrowband_distances_cuda_batch(inputs);
}

void evaluate_narrowband_distances_cuda_batch(
    const std::vector<NarrowbandEvaluationInput> &inputs) {
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  size_t fragment_count = 0;
  size_t candidate_count = 0;
  for (const NarrowbandEvaluationInput &input : inputs) {
    if (!input.mesh || !input.fragments || !input.candidates ||
        !input.distances) {
      throw std::invalid_argument(
          "narrowband CUDA batch contains a null input");
    }
    if (!std::isfinite(input.scale) || input.scale <= 0.0 ||
        !std::isfinite(input.voxel_size) || input.voxel_size <= 0.0) {
      throw std::invalid_argument(
          "narrowband CUDA scale and voxel size must be positive");
    }
    if (input.mesh->vertices.empty() || input.mesh->triangles.empty())
      throw std::invalid_argument("narrowband CUDA mesh is empty");
    if (input.mesh->vertices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                vertex_count ||
        input.mesh->triangles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) -
                triangle_count) {
      throw std::overflow_error("narrowband CUDA packed index overflow");
    }
    vertex_count += input.mesh->vertices.size();
    triangle_count += input.mesh->triangles.size();
    if (input.fragments->size() >
            std::numeric_limits<size_t>::max() - fragment_count ||
        input.candidates->size() >
            std::numeric_limits<size_t>::max() - candidate_count) {
      throw std::overflow_error("narrowband CUDA packed size overflow");
    }
    fragment_count += input.fragments->size();
    candidate_count += input.candidates->size();
    input.distances->resize(input.candidates->size());
  }
  if (candidate_count == 0)
    return;

  Runtime &state = runtime();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_stream();
  state.vertices.ensure(vertex_count * sizeof(float3),
                        "allocate narrowband vertices");
  state.triangles.ensure(triangle_count * sizeof(int3),
                         "allocate narrowband triangles");
  state.fragments.ensure(fragment_count * sizeof(NarrowbandFragment),
                         "allocate narrowband fragments");
  state.candidates.ensure(candidate_count * sizeof(PackedNarrowbandCandidate),
                          "allocate narrowband candidates");
  state.distances.ensure(candidate_count * sizeof(NarrowbandDistance),
                         "allocate narrowband distances");
  state.host_vertices.ensure(vertex_count * sizeof(float3),
                             "allocate host narrowband vertices");
  state.host_triangles.ensure(triangle_count * sizeof(int3),
                              "allocate host narrowband triangles");
  state.host_fragments.ensure(
      fragment_count * sizeof(NarrowbandFragment),
      "allocate host narrowband fragments");
  state.host_candidates.ensure(
      candidate_count * sizeof(PackedNarrowbandCandidate),
      "allocate host narrowband candidates");
  state.host_distances.ensure(
      candidate_count * sizeof(NarrowbandDistance),
      "allocate host narrowband distances");

  float3 *host_vertices = state.host_vertices.as<float3>();
  int3 *host_triangles = state.host_triangles.as<int3>();
  NarrowbandFragment *host_fragments =
      state.host_fragments.as<NarrowbandFragment>();
  PackedNarrowbandCandidate *host_candidates =
      state.host_candidates.as<PackedNarrowbandCandidate>();
  size_t vertex_offset = 0;
  size_t triangle_offset = 0;
  size_t fragment_offset = 0;
  size_t candidate_offset = 0;
  for (const NarrowbandEvaluationInput &input : inputs) {
    const Mesh &mesh = *input.mesh;
    for (size_t index = 0; index < mesh.vertices.size(); ++index) {
      const Vec3D &vertex = mesh.vertices[index];
      host_vertices[vertex_offset + index] = make_float3(
          static_cast<float>(vertex[0] * input.scale),
          static_cast<float>(vertex[1] * input.scale),
          static_cast<float>(vertex[2] * input.scale));
    }
    for (size_t index = 0; index < mesh.triangles.size(); ++index) {
      const auto &triangle = mesh.triangles[index];
      host_triangles[triangle_offset + index] = make_int3(
          static_cast<int>(vertex_offset) + triangle[0],
          static_cast<int>(vertex_offset) + triangle[1],
          static_cast<int>(vertex_offset) + triangle[2]);
    }
    for (size_t index = 0; index < input.fragments->size(); ++index) {
      NarrowbandFragment fragment = (*input.fragments)[index];
      fragment.triangle_index += static_cast<int>(triangle_offset);
      host_fragments[fragment_offset + index] = fragment;
    }
    for (size_t index = 0; index < input.candidates->size(); ++index) {
      const NarrowbandCandidate &candidate = (*input.candidates)[index];
      host_candidates[candidate_offset + index] = {
          candidate.x, candidate.y, candidate.z,
          candidate.manhattan_limit,
          fragment_offset + candidate.fragment_offset,
          candidate.fragment_count, static_cast<int>(triangle_offset),
          input.voxel_size};
    }
    vertex_offset += mesh.vertices.size();
    triangle_offset += mesh.triangles.size();
    fragment_offset += input.fragments->size();
    candidate_offset += input.candidates->size();
  }

  cuda_memory::check(
      cudaMemcpyAsync(state.vertices.as<float3>(), host_vertices,
                      vertex_count * sizeof(float3),
                      cudaMemcpyHostToDevice, state.stream),
      "copy narrowband vertices");
  cuda_memory::check(
      cudaMemcpyAsync(state.triangles.as<int3>(), host_triangles,
                      triangle_count * sizeof(int3),
                      cudaMemcpyHostToDevice, state.stream),
      "copy narrowband triangles");
  cuda_memory::check(
      cudaMemcpyAsync(state.fragments.as<NarrowbandFragment>(),
                      state.host_fragments.as<NarrowbandFragment>(),
                      fragment_count * sizeof(NarrowbandFragment),
                      cudaMemcpyHostToDevice, state.stream),
      "copy narrowband fragments");
  cuda_memory::check(
      cudaMemcpyAsync(state.candidates.as<PackedNarrowbandCandidate>(),
                      state.host_candidates.as<PackedNarrowbandCandidate>(),
                      candidate_count * sizeof(PackedNarrowbandCandidate),
                      cudaMemcpyHostToDevice, state.stream),
      "copy narrowband candidates");

  constexpr int threads = 128;
  const int blocks = static_cast<int>(
      (candidate_count + threads - 1) / threads);
  evaluate_narrowband_kernel<<<blocks, threads, 0, state.stream>>>(
      state.vertices.as<float3>(), state.triangles.as<int3>(),
      state.fragments.as<NarrowbandFragment>(),
      state.candidates.as<PackedNarrowbandCandidate>(), candidate_count,
      state.distances.as<NarrowbandDistance>());
  cuda_memory::check(cudaGetLastError(),
                     "launch narrowband distance evaluation");
  cuda_memory::check(
      cudaMemcpyAsync(state.host_distances.as<NarrowbandDistance>(),
                      state.distances.as<NarrowbandDistance>(),
                      candidate_count * sizeof(NarrowbandDistance),
                      cudaMemcpyDeviceToHost, state.stream),
      "copy narrowband distances");
  cuda_memory::check(cudaStreamSynchronize(state.stream),
                     "wait for narrowband distance evaluation");
  candidate_offset = 0;
  for (const NarrowbandEvaluationInput &input : inputs) {
    std::copy_n(state.host_distances.as<NarrowbandDistance>() +
                    candidate_offset,
                input.candidates->size(), input.distances->begin());
    candidate_offset += input.candidates->size();
  }
}

} // namespace neural_acd
