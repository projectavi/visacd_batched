#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <config.hpp>
#include <convex_hull_batch.hpp>
#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <device_mesh.hpp>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
  if (first && second > std::numeric_limits<size_t>::max() / first)
    throw std::overflow_error(message);
  return first * second;
}

using cuda_memory::DeviceBuffer;
using cuda_memory::PinnedBuffer;

struct PackedAssignmentJob {
  const double3 *vertices;
  const double4 *faces;
  int vertex_count;
  int face_count;
  double epsilon;
};

__device__ double signed_distance(const double4 &plane,
                                  const double3 &point) {
  return plane.x * point.x + plane.y * point.y +
         plane.z * point.z + plane.w;
}

__global__ void assign_points_kernel(
    const PackedAssignmentJob *jobs, const int *candidate_jobs,
    const int *candidate_vertices, int *assignments, double *distances,
    int candidate_count) {
  const int candidate_index =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate_index >= candidate_count)
    return;
  const PackedAssignmentJob job = jobs[candidate_jobs[candidate_index]];
  const int vertex_index = candidate_vertices[candidate_index];
  int assignment = -1;
  double selected_distance = 0.0;
  if (vertex_index >= 0 && vertex_index < job.vertex_count) {
    const double3 point = job.vertices[vertex_index];
    for (int face_index = 0; face_index < job.face_count; ++face_index) {
      const double distance = signed_distance(job.faces[face_index], point);
      if (distance > job.epsilon) {
        assignment = face_index;
        selected_distance = distance;
        break;
      }
    }
  }
  assignments[candidate_index] = assignment;
  distances[candidate_index] = selected_distance;
}

struct DeviceTopologyFace {
  int3 vertices;
  double4 plane;
  int farthest;
  double farthest_distance;
  int outside_count;
  int active;
};

struct PackedTopologyJob {
  const double3 *vertices;
  DeviceTopologyFace *faces;
  int *assignments;
  double *distances;
  unsigned char *candidates;
  unsigned char *visible;
  int *pending_faces;
  int *horizon_edges;
  int3 *output_triangles;
  int *output_count;
  int *status;
  int vertex_count;
  int face_capacity;
  int output_capacity;
};

__device__ double3 topology_subtract(double3 first, double3 second) {
  return make_double3(first.x - second.x, first.y - second.y,
                      first.z - second.z);
}

__device__ double3 topology_cross(double3 first, double3 second) {
  return make_double3(first.y * second.z - first.z * second.y,
                      first.z * second.x - first.x * second.z,
                      first.x * second.y - first.y * second.x);
}

__device__ double topology_dot(double3 first, double3 second) {
  return first.x * second.x + first.y * second.y + first.z * second.z;
}

__device__ double topology_squared_length(double3 value) {
  return topology_dot(value, value);
}

__device__ unsigned long long topology_edge_key(int first, int second) {
  const unsigned int low = static_cast<unsigned int>(min(first, second));
  const unsigned int high = static_cast<unsigned int>(max(first, second));
  return (static_cast<unsigned long long>(low) << 32) | high;
}

__device__ int topology_append_face(const PackedTopologyJob &job,
                                    int &face_count, int first, int second,
                                    int third, double3 interior,
                                    double epsilon) {
  if (face_count >= job.face_capacity || first == second || first == third ||
      second == third)
    return -1;
  const double3 a = job.vertices[first];
  double3 b = job.vertices[second];
  double3 c = job.vertices[third];
  double3 normal = topology_cross(topology_subtract(b, a),
                                  topology_subtract(c, a));
  double length = sqrt(topology_squared_length(normal));
  if (!(length > epsilon))
    return -1;
  if (topology_dot(normal, topology_subtract(interior, a)) > 0.0) {
    const int swapped = second;
    second = third;
    third = swapped;
    b = job.vertices[second];
    c = job.vertices[third];
    normal = topology_cross(topology_subtract(b, a),
                            topology_subtract(c, a));
    length = sqrt(topology_squared_length(normal));
  }
  normal.x /= length;
  normal.y /= length;
  normal.z /= length;
  DeviceTopologyFace &face = job.faces[face_count];
  face.vertices = make_int3(first, second, third);
  face.plane = make_double4(normal.x, normal.y, normal.z,
                            -topology_dot(normal, a));
  face.farthest = -1;
  face.farthest_distance = 0.0;
  face.outside_count = 0;
  face.active = 1;
  return face_count++;
}

__device__ bool topology_initialize(
    const PackedTopologyJob &job, int &face_count, int &pending_head,
    int &pending_tail, double3 &interior, double &epsilon,
    int &classify_begin, int &classify_count) {
  if (job.vertex_count < 4)
    return false;
  int extremes[6] = {0, 0, 0, 0, 0, 0};
  double3 minimum = job.vertices[0];
  double3 maximum = minimum;
  for (int index = 1; index < job.vertex_count; ++index) {
    const double3 point = job.vertices[index];
    const double coordinates[3] = {point.x, point.y, point.z};
    double *minimum_coordinates = &minimum.x;
    double *maximum_coordinates = &maximum.x;
    for (int axis = 0; axis < 3; ++axis) {
      if (coordinates[axis] < minimum_coordinates[axis]) {
        minimum_coordinates[axis] = coordinates[axis];
        extremes[axis * 2] = index;
      }
      if (coordinates[axis] > maximum_coordinates[axis]) {
        maximum_coordinates[axis] = coordinates[axis];
        extremes[axis * 2 + 1] = index;
      }
    }
  }
  const double scale = max(
      max(maximum.x - minimum.x, maximum.y - minimum.y),
      max(maximum.z - minimum.z, 1.0));
  epsilon = scale * 1e-12;

  int first = -1;
  int second = -1;
  double best_pair_distance = -1.0;
  for (int a = 0; a < 6; ++a) {
    for (int b = a + 1; b < 6; ++b) {
      const double distance = topology_squared_length(topology_subtract(
          job.vertices[extremes[b]], job.vertices[extremes[a]]));
      if (distance > best_pair_distance) {
        best_pair_distance = distance;
        first = extremes[a];
        second = extremes[b];
      }
    }
  }
  if (!(best_pair_distance > epsilon * epsilon))
    return false;

  const double3 line =
      topology_subtract(job.vertices[second], job.vertices[first]);
  int third = -1;
  double best_line_distance = -1.0;
  for (int index = 0; index < job.vertex_count; ++index) {
    if (index == first || index == second)
      continue;
    const double distance = topology_squared_length(topology_cross(
        topology_subtract(job.vertices[index], job.vertices[first]), line));
    if (distance > best_line_distance) {
      best_line_distance = distance;
      third = index;
    }
  }
  if (!(best_line_distance > epsilon * epsilon * best_pair_distance))
    return false;

  const double3 base_normal = topology_cross(
      topology_subtract(job.vertices[second], job.vertices[first]),
      topology_subtract(job.vertices[third], job.vertices[first]));
  const double normal_length = sqrt(topology_squared_length(base_normal));
  int fourth = -1;
  double best_plane_distance = -1.0;
  for (int index = 0; index < job.vertex_count; ++index) {
    if (index == first || index == second || index == third)
      continue;
    const double distance = fabs(topology_dot(
                                base_normal,
                                topology_subtract(job.vertices[index],
                                                  job.vertices[first]))) /
                            normal_length;
    if (distance > best_plane_distance) {
      best_plane_distance = distance;
      fourth = index;
    }
  }
  if (!(best_plane_distance > epsilon))
    return false;

  interior = make_double3(
      (job.vertices[first].x + job.vertices[second].x +
       job.vertices[third].x + job.vertices[fourth].x) /
          4.0,
      (job.vertices[first].y + job.vertices[second].y +
       job.vertices[third].y + job.vertices[fourth].y) /
          4.0,
      (job.vertices[first].z + job.vertices[second].z +
       job.vertices[third].z + job.vertices[fourth].z) /
          4.0);
  if (topology_append_face(job, face_count, first, second, third, interior,
                           epsilon) < 0 ||
      topology_append_face(job, face_count, first, fourth, second, interior,
                           epsilon) < 0 ||
      topology_append_face(job, face_count, second, fourth, third, interior,
                           epsilon) < 0 ||
      topology_append_face(job, face_count, third, fourth, first, interior,
                           epsilon) < 0) {
    return false;
  }
  for (int index = 0; index < job.vertex_count; ++index) {
    job.assignments[index] = -1;
    job.candidates[index] =
        index != first && index != second && index != third &&
        index != fourth;
  }
  pending_head = 0;
  pending_tail = 0;
  classify_begin = 0;
  classify_count = 4;
  return true;
}

__device__ bool topology_select_expansion(
    const PackedTopologyJob &job, int face_count, int &pending_head,
    int pending_tail, int &expansions, int &selected_index,
    int &active_vertex) {
  while (pending_head < pending_tail) {
    selected_index = job.pending_faces[pending_head++];
    if (selected_index < 0 || selected_index >= face_count)
      return false;
    DeviceTopologyFace &selected = job.faces[selected_index];
    if (!selected.active || selected.farthest < 0 ||
        selected.outside_count == 0) {
      continue;
    }
    if (++expansions > job.vertex_count)
      return false;
    active_vertex = selected.farthest;
    return true;
  }
  selected_index = -1;
  active_vertex = -1;
  return true;
}

__device__ bool topology_build_horizon(
    const PackedTopologyJob &job, int &face_count, int active_vertex,
    double3 interior, double epsilon,
    int &classify_begin, int &classify_count) {
  const int horizon_begin = face_count;
  for (int face_index = 0; face_index < horizon_begin; ++face_index) {
    if (!job.visible[face_index])
      continue;
    const int3 vertices = job.faces[face_index].vertices;
    const int face_vertices[3] = {vertices.x, vertices.y, vertices.z};
    for (int edge = 0; edge < 3; ++edge) {
      const int first = face_vertices[edge];
      const int second = face_vertices[(edge + 1) % 3];
      if (job.horizon_edges[face_index * 3 + edge] &&
          topology_append_face(job, face_count, first, second,
                               active_vertex, interior, epsilon) < 0) {
        return false;
      }
    }
    job.faces[face_index].active = 0;
    job.faces[face_index].outside_count = 0;
    job.faces[face_index].farthest = -1;
  }
  classify_begin = horizon_begin;
  classify_count = face_count - horizon_begin;
  return classify_count >= 3;
}

__global__ void build_hull_topology_kernel(PackedTopologyJob *jobs,
                                           int job_count) {
  const int job_index = blockIdx.x;
  if (job_index >= job_count)
    return;
  const PackedTopologyJob job = jobs[job_index];
  __shared__ int face_count;
  __shared__ int pending_head;
  __shared__ int pending_tail;
  __shared__ int expansions;
  __shared__ int classify_begin;
  __shared__ int classify_count;
  __shared__ int selected_index;
  __shared__ int active_vertex;
  __shared__ int visible_count;
  __shared__ int action;
  __shared__ int invalid;
  __shared__ double3 interior;
  __shared__ double epsilon;

  if (threadIdx.x == 0) {
    face_count = 0;
    expansions = 0;
    action = topology_initialize(job, face_count, pending_head, pending_tail,
                                 interior, epsilon, classify_begin,
                                 classify_count)
                 ? 1
                 : -1;
  }
  __syncthreads();

  while (action == 1) {
    for (int vertex = threadIdx.x; vertex < job.vertex_count;
         vertex += blockDim.x) {
      if (!job.candidates[vertex])
        continue;
      int assignment = -1;
      double distance = 0.0;
      const double3 point = job.vertices[vertex];
      for (int local_face = 0; local_face < classify_count; ++local_face) {
        const int face_index = classify_begin + local_face;
        const double candidate =
            signed_distance(job.faces[face_index].plane, point);
        if (candidate > epsilon) {
          assignment = face_index;
          distance = candidate;
          break;
        }
      }
      job.assignments[vertex] = assignment;
      job.distances[vertex] = distance;
    }
    __syncthreads();

    for (int face = classify_begin + threadIdx.x;
         face < classify_begin + classify_count; face += blockDim.x) {
      job.faces[face].farthest = INT_MAX;
      job.faces[face].farthest_distance = 0.0;
      job.faces[face].outside_count = 0;
    }
    __syncthreads();

    for (int vertex = threadIdx.x; vertex < job.vertex_count;
         vertex += blockDim.x) {
      if (!job.candidates[vertex])
        continue;
      const int assignment = job.assignments[vertex];
      if (assignment < classify_begin ||
          assignment >= classify_begin + classify_count) {
        continue;
      }
      DeviceTopologyFace &face = job.faces[assignment];
      atomicAdd(&face.outside_count, 1);
      atomicMax(reinterpret_cast<unsigned long long *>(
                    &face.farthest_distance),
                static_cast<unsigned long long>(
                    __double_as_longlong(job.distances[vertex])));
    }
    __syncthreads();

    for (int vertex = threadIdx.x; vertex < job.vertex_count;
         vertex += blockDim.x) {
      if (!job.candidates[vertex])
        continue;
      const int assignment = job.assignments[vertex];
      if (assignment < classify_begin ||
          assignment >= classify_begin + classify_count) {
        continue;
      }
      DeviceTopologyFace &face = job.faces[assignment];
      if (__double_as_longlong(job.distances[vertex]) ==
          __double_as_longlong(face.farthest_distance)) {
        atomicMin(&face.farthest, vertex);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      for (int face = classify_begin;
           face < classify_begin + classify_count; ++face) {
        if (job.faces[face].outside_count > 0) {
          if (pending_tail >= job.face_capacity) {
            action = -1;
            break;
          }
          job.pending_faces[pending_tail++] = face;
        } else {
          job.faces[face].farthest = -1;
        }
      }
      if (action != -1) {
        if (!topology_select_expansion(
                job, face_count, pending_head, pending_tail, expansions,
                selected_index, active_vertex)) {
          action = -1;
        } else {
          action = selected_index >= 0 ? 2 : 0;
          visible_count = 0;
        }
      }
    }
    __syncthreads();

    if (action != 2)
      continue;

    const double3 active_point = job.vertices[active_vertex];
    for (int face = threadIdx.x; face < face_count; face += blockDim.x) {
      const bool visible = job.faces[face].active &&
          signed_distance(job.faces[face].plane, active_point) > epsilon;
      job.visible[face] = visible;
      if (visible)
        atomicAdd(&visible_count, 1);
    }
    __syncthreads();

    for (int vertex = threadIdx.x; vertex < job.vertex_count;
         vertex += blockDim.x) {
      const int assignment = job.assignments[vertex];
      const bool from_visible_face =
          assignment >= 0 && assignment < face_count &&
          job.visible[assignment];
      job.candidates[vertex] =
          from_visible_face && vertex != active_vertex;
      if (from_visible_face)
        job.assignments[vertex] = -1;
    }
    __syncthreads();

    for (int face_edge = threadIdx.x; face_edge < face_count * 3;
         face_edge += blockDim.x) {
      const int face_index = face_edge / 3;
      const int edge = face_edge % 3;
      int is_horizon = 0;
      if (job.visible[face_index]) {
        const int3 vertices = job.faces[face_index].vertices;
        const int face_vertices[3] = {vertices.x, vertices.y, vertices.z};
        const unsigned long long key = topology_edge_key(
            face_vertices[edge], face_vertices[(edge + 1) % 3]);
        int edge_count = 0;
        for (int other_face = 0;
             other_face < face_count && edge_count < 2; ++other_face) {
          if (!job.visible[other_face])
            continue;
          const int3 other_vertices = job.faces[other_face].vertices;
          const int values[3] = {other_vertices.x, other_vertices.y,
                                 other_vertices.z};
          for (int other_edge = 0; other_edge < 3; ++other_edge) {
            if (topology_edge_key(values[other_edge],
                                  values[(other_edge + 1) % 3]) == key) {
              ++edge_count;
              break;
            }
          }
        }
        is_horizon = edge_count == 1;
      }
      job.horizon_edges[face_edge] = is_horizon;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      if (!job.visible[selected_index] || visible_count == 0) {
        action = -1;
      } else {
        action = topology_build_horizon(
                     job, face_count, active_vertex, interior, epsilon,
                     classify_begin, classify_count)
                     ? 1
                     : -1;
      }
    }
    __syncthreads();
  }

  if (action < 0) {
    if (threadIdx.x == 0)
      *job.status = -1;
    return;
  }

  if (threadIdx.x == 0)
    invalid = 0;
  __syncthreads();
  for (int vertex = threadIdx.x; vertex < job.vertex_count;
       vertex += blockDim.x) {
    const double3 point = job.vertices[vertex];
    for (int face = 0; face < face_count; ++face) {
      if (job.faces[face].active &&
          signed_distance(job.faces[face].plane, point) > epsilon) {
        atomicExch(&invalid, 1);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int output_count = 0;
    for (int face = 0; face < face_count; ++face) {
      if (!job.faces[face].active)
        continue;
      if (output_count >= job.output_capacity) {
        invalid = 1;
        break;
      }
      job.output_triangles[output_count++] = job.faces[face].vertices;
    }
    *job.output_count = output_count;
    *job.status = !invalid && output_count >= 4 ? face_count : -1;
  }
}

double squared_length(const Vec3D &value) {
  return value[0] * value[0] + value[1] * value[1] +
         value[2] * value[2];
}

double dot_product(const Vec3D &first, const Vec3D &second) {
  return first[0] * second[0] + first[1] * second[1] +
         first[2] * second[2];
}

Vec3D cross(const Vec3D &first, const Vec3D &second) {
  return {first[1] * second[2] - first[2] * second[1],
          first[2] * second[0] - first[0] * second[2],
          first[0] * second[1] - first[1] * second[0]};
}

struct HullFace {
  std::array<int, 3> vertices{};
  double4 plane{};
  std::vector<int> outside;
  int farthest = -1;
  double farthest_distance = 0.0;
  bool active = true;
};

struct HullState {
  ConvexHullBatchInput input;
  std::vector<HullFace> faces;
  std::deque<int> pending_faces;
  Vec3D interior{};
  double epsilon = 0.0;
  size_t expansions = 0;
  bool failed = false;
};

struct AssignmentRequest {
  HullState *state = nullptr;
  std::vector<int> candidates;
  std::vector<int> faces;
  bool validation = false;
};

uint64_t edge_key(int first, int second) {
  const uint32_t low =
      static_cast<uint32_t>(std::min(first, second));
  const uint32_t high =
      static_cast<uint32_t>(std::max(first, second));
  return (static_cast<uint64_t>(low) << 32) | high;
}

double face_distance(const HullFace &face, const Vec3D &point) {
  return face.plane.x * point[0] + face.plane.y * point[1] +
         face.plane.z * point[2] + face.plane.w;
}

int append_face(HullState &state, int first, int second, int third) {
  const Mesh &mesh = *state.input.mesh;
  if (first == second || first == third || second == third)
    return -1;
  const Vec3D &a = mesh.vertices[first];
  const Vec3D &b = mesh.vertices[second];
  const Vec3D &c = mesh.vertices[third];
  Vec3D normal = cross(b - a, c - a);
  double length = std::sqrt(squared_length(normal));
  if (!(length > state.epsilon))
    return -1;
  if (dot_product(normal, state.interior - a) > 0.0) {
    std::swap(second, third);
    const Vec3D &flipped_b = mesh.vertices[second];
    const Vec3D &flipped_c = mesh.vertices[third];
    normal = cross(flipped_b - a, flipped_c - a);
    length = std::sqrt(squared_length(normal));
  }
  normal = normal / length;
  HullFace face;
  face.vertices = {first, second, third};
  face.plane =
      make_double4(normal[0], normal[1], normal[2],
                   -dot_product(normal, a));
  state.faces.push_back(std::move(face));
  return static_cast<int>(state.faces.size() - 1);
}

bool initialize_hull(HullState &state, AssignmentRequest &request) {
  const Mesh &mesh = *state.input.mesh;
  const size_t count = mesh.vertices.size();
  if (count < 4 ||
      count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  std::array<int, 6> extremes{0, 0, 0, 0, 0, 0};
  Vec3D minimum = mesh.vertices[0];
  Vec3D maximum = mesh.vertices[0];
  for (size_t index = 1; index < count; ++index) {
    const Vec3D &point = mesh.vertices[index];
    for (int axis = 0; axis < 3; ++axis) {
      if (point[axis] < minimum[axis]) {
        minimum[axis] = point[axis];
        extremes[axis * 2] = static_cast<int>(index);
      }
      if (point[axis] > maximum[axis]) {
        maximum[axis] = point[axis];
        extremes[axis * 2 + 1] = static_cast<int>(index);
      }
    }
  }
  const double scale =
      std::max({maximum[0] - minimum[0], maximum[1] - minimum[1],
                maximum[2] - minimum[2], 1.0});
  state.epsilon = scale * 1e-12;

  int first = -1;
  int second = -1;
  double best_pair_distance = -1.0;
  for (size_t a = 0; a < extremes.size(); ++a) {
    for (size_t b = a + 1; b < extremes.size(); ++b) {
      const double distance = squared_length(
          mesh.vertices[extremes[b]] - mesh.vertices[extremes[a]]);
      if (distance > best_pair_distance) {
        best_pair_distance = distance;
        first = extremes[a];
        second = extremes[b];
      }
    }
  }
  if (!(best_pair_distance > state.epsilon * state.epsilon))
    return false;

  const Vec3D line = mesh.vertices[second] - mesh.vertices[first];
  int third = -1;
  double best_line_distance = -1.0;
  for (size_t index = 0; index < count; ++index) {
    if (static_cast<int>(index) == first ||
        static_cast<int>(index) == second) {
      continue;
    }
    const double distance = squared_length(
        cross(mesh.vertices[index] - mesh.vertices[first], line));
    if (distance > best_line_distance) {
      best_line_distance = distance;
      third = static_cast<int>(index);
    }
  }
  if (!(best_line_distance >
        state.epsilon * state.epsilon * best_pair_distance)) {
    return false;
  }

  const Vec3D base_normal =
      cross(mesh.vertices[second] - mesh.vertices[first],
            mesh.vertices[third] - mesh.vertices[first]);
  const double normal_length = std::sqrt(squared_length(base_normal));
  int fourth = -1;
  double best_plane_distance = -1.0;
  for (size_t index = 0; index < count; ++index) {
    if (static_cast<int>(index) == first ||
        static_cast<int>(index) == second ||
        static_cast<int>(index) == third) {
      continue;
    }
    const double distance =
        std::abs(dot_product(base_normal,
                             mesh.vertices[index] - mesh.vertices[first])) /
        normal_length;
    if (distance > best_plane_distance) {
      best_plane_distance = distance;
      fourth = static_cast<int>(index);
    }
  }
  if (!(best_plane_distance > state.epsilon))
    return false;

  state.interior =
      (mesh.vertices[first] + mesh.vertices[second] +
       mesh.vertices[third] + mesh.vertices[fourth]) /
      4.0;
  const int face0 = append_face(state, first, second, third);
  const int face1 = append_face(state, first, fourth, second);
  const int face2 = append_face(state, second, fourth, third);
  const int face3 = append_face(state, third, fourth, first);
  if (face0 < 0 || face1 < 0 || face2 < 0 || face3 < 0)
    return false;

  request.state = &state;
  request.faces = {face0, face1, face2, face3};
  request.candidates.reserve(count - 4);
  for (size_t index = 0; index < count; ++index) {
    const int vertex = static_cast<int>(index);
    if (vertex != first && vertex != second && vertex != third &&
        vertex != fourth) {
      request.candidates.push_back(vertex);
    }
  }
  return true;
}

void scatter_assignments(AssignmentRequest &request, const int *assignments,
                         const double *distances) {
  HullState &state = *request.state;
  if (request.validation) {
    for (size_t index = 0; index < request.candidates.size(); ++index) {
      if (assignments[index] >= 0) {
        state.failed = true;
        return;
      }
    }
    return;
  }
  for (size_t index = 0; index < request.candidates.size(); ++index) {
    const int local_face = assignments[index];
    if (local_face < 0)
      continue;
    if (local_face >= static_cast<int>(request.faces.size())) {
      state.failed = true;
      return;
    }
    HullFace &face = state.faces[request.faces[local_face]];
    const int vertex = request.candidates[index];
    face.outside.push_back(vertex);
    if (face.farthest < 0 ||
        distances[index] > face.farthest_distance) {
      face.farthest = vertex;
      face.farthest_distance = distances[index];
    }
  }
  for (int face_index : request.faces) {
    if (!state.faces[face_index].outside.empty())
      state.pending_faces.push_back(face_index);
  }
}

struct OrientedEdge {
  uint64_t key;
  int first;
  int second;
};

bool prepare_expansion(HullState &state, AssignmentRequest &request) {
  const Mesh &mesh = *state.input.mesh;
  while (!state.pending_faces.empty()) {
    const int selected_index = state.pending_faces.front();
    state.pending_faces.pop_front();
    if (selected_index < 0 ||
        selected_index >= static_cast<int>(state.faces.size())) {
      state.failed = true;
      return false;
    }
    HullFace &selected = state.faces[selected_index];
    if (!selected.active || selected.farthest < 0 ||
        selected.outside.empty()) {
      continue;
    }
    if (++state.expansions > mesh.vertices.size()) {
      state.failed = true;
      return false;
    }

    const int active_vertex = selected.farthest;
    const Vec3D &active_point = mesh.vertices[active_vertex];
    std::vector<unsigned char> visible(state.faces.size(), 0);
    size_t visible_count = 0;
    for (size_t face_index = 0; face_index < state.faces.size();
         ++face_index) {
      const HullFace &face = state.faces[face_index];
      if (face.active &&
          face_distance(face, active_point) > state.epsilon) {
        visible[face_index] = 1;
        ++visible_count;
      }
    }
    if (!visible[selected_index] || visible_count == 0) {
      state.failed = true;
      return false;
    }

    std::unordered_map<uint64_t, int> visible_edge_counts;
    std::vector<OrientedEdge> visible_edges;
    std::vector<int> candidates;
    for (size_t face_index = 0; face_index < state.faces.size();
         ++face_index) {
      if (!visible[face_index])
        continue;
      HullFace &face = state.faces[face_index];
      for (int vertex : face.outside) {
        if (vertex != active_vertex)
          candidates.push_back(vertex);
      }
      for (int edge = 0; edge < 3; ++edge) {
        const int first = face.vertices[edge];
        const int second = face.vertices[(edge + 1) % 3];
        const uint64_t key = edge_key(first, second);
        ++visible_edge_counts[key];
        visible_edges.push_back({key, first, second});
      }
      face.active = false;
      face.outside.clear();
      face.farthest = -1;
    }

    std::vector<OrientedEdge> horizon;
    std::unordered_set<uint64_t> added_horizon_edges;
    for (const OrientedEdge &edge : visible_edges) {
      if (visible_edge_counts[edge.key] == 1 &&
          added_horizon_edges.insert(edge.key).second) {
        horizon.push_back(edge);
      }
    }
    if (horizon.size() < 3) {
      state.failed = true;
      return false;
    }

    request = {};
    request.state = &state;
    request.faces.reserve(horizon.size());
    for (const OrientedEdge &edge : horizon) {
      const int face_index =
          append_face(state, edge.first, edge.second, active_vertex);
      if (face_index < 0) {
        state.failed = true;
        return false;
      }
      request.faces.push_back(face_index);
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    request.candidates = std::move(candidates);
    if (!request.candidates.empty())
      return true;
  }
  return false;
}

bool assemble_hull(HullState &state) {
  std::unordered_map<uint64_t, int> edge_counts;
  size_t triangle_count = 0;
  for (const HullFace &face : state.faces) {
    if (!face.active)
      continue;
    ++triangle_count;
    for (int edge = 0; edge < 3; ++edge) {
      ++edge_counts[edge_key(face.vertices[edge],
                             face.vertices[(edge + 1) % 3])];
    }
  }
  if (triangle_count < 4)
    return false;
  for (const auto &entry : edge_counts) {
    if (entry.second != 2)
      return false;
  }

  Mesh &hull = *state.input.hull;
  hull.clear();
  hull.cut_verts.clear();
  hull.is_new.clear();
  hull.intersecting_edges.clear();
  hull.vertices = state.input.mesh->vertices;
  hull.triangles.reserve(triangle_count);
  for (const HullFace &face : state.faces) {
    if (face.active)
      hull.triangles.push_back(face.vertices);
  }
  if (state.input.fix_normals)
    cvx_fix_normals(hull);
  return true;
}

void compute_fallback(const ConvexHullBatchInput &input) {
  input.hull->clear();
  input.hull->cut_verts.clear();
  input.hull->is_new.clear();
  input.hull->intersecting_edges.clear();
  input.mesh->compute_ch(*input.hull, input.fix_normals);
}

size_t growth_bytes(const DeviceBuffer &buffer, size_t requested) {
  return requested > buffer.capacity() ? requested - buffer.capacity() : 0;
}

} // namespace

struct ConvexHullBatchRuntime::Impl {
  cudaStream_t stream = nullptr;
  DeviceBuffer vertices;
  DeviceBuffer faces;
  DeviceBuffer jobs;
  DeviceBuffer candidate_jobs;
  DeviceBuffer candidate_vertices;
  DeviceBuffer assignments;
  DeviceBuffer distances;
  DeviceBuffer topology_faces;
  DeviceBuffer topology_jobs;
  DeviceBuffer topology_assignments;
  DeviceBuffer topology_distances;
  DeviceBuffer topology_candidates;
  DeviceBuffer topology_visible;
  DeviceBuffer topology_pending_faces;
  DeviceBuffer topology_horizon_edges;
  DeviceBuffer topology_output_triangles;
  DeviceBuffer topology_output_counts;
  DeviceBuffer topology_status;
  PinnedBuffer host_vertices;
  PinnedBuffer host_faces;
  PinnedBuffer host_jobs;
  PinnedBuffer host_candidate_jobs;
  PinnedBuffer host_candidate_vertices;
  PinnedBuffer host_assignments;
  PinnedBuffer host_distances;
  PinnedBuffer host_topology_jobs;
  PinnedBuffer host_topology_output_triangles;
  PinnedBuffer host_topology_output_counts;
  PinnedBuffer host_topology_status;

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (!stream) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags hull");
    }
    DeviceBuffer::set_allocation_stream(stream);
  }

  size_t growth(size_t vertex_count, size_t face_count,
                size_t candidate_count, size_t job_count) const {
    size_t result = 0;
    const auto include = [&](const DeviceBuffer &buffer, size_t count,
                             size_t element_size) {
      result = checked_add(
          result,
          growth_bytes(buffer,
                       checked_multiply(count, element_size,
                                        "Hull allocation overflow")),
          "Hull allocation total overflow");
    };
    include(vertices, vertex_count, sizeof(double3));
    include(faces, face_count, sizeof(double4));
    include(jobs, job_count, sizeof(PackedAssignmentJob));
    include(candidate_jobs, candidate_count, sizeof(int));
    include(candidate_vertices, candidate_count, sizeof(int));
    include(assignments, candidate_count, sizeof(int));
    include(distances, candidate_count, sizeof(double));
    return result;
  }

  size_t topology_growth(size_t packed_vertex_count, size_t vertex_count,
                         size_t face_count, size_t horizon_edge_count,
                         size_t output_count, size_t job_count) const {
    size_t result = 0;
    const auto include = [&](const DeviceBuffer &buffer, size_t count,
                             size_t element_size) {
      result = checked_add(
          result,
          growth_bytes(buffer,
                       checked_multiply(count, element_size,
                                        "Hull topology allocation overflow")),
          "Hull topology allocation total overflow");
    };
    include(vertices, packed_vertex_count, sizeof(double3));
    include(topology_faces, face_count, sizeof(DeviceTopologyFace));
    include(topology_jobs, job_count, sizeof(PackedTopologyJob));
    include(topology_assignments, vertex_count, sizeof(int));
    include(topology_distances, vertex_count, sizeof(double));
    include(topology_candidates, vertex_count, sizeof(unsigned char));
    include(topology_visible, face_count, sizeof(unsigned char));
    include(topology_pending_faces, face_count, sizeof(int));
    include(topology_horizon_edges, horizon_edge_count, sizeof(int));
    include(topology_output_triangles, output_count, sizeof(int3));
    include(topology_output_counts, job_count, sizeof(int));
    include(topology_status, job_count, sizeof(int));
    return result;
  }
};

ConvexHullBatchRuntime::ConvexHullBatchRuntime()
    : impl_(std::make_unique<Impl>()) {}
ConvexHullBatchRuntime::~ConvexHullBatchRuntime() = default;
ConvexHullBatchRuntime::ConvexHullBatchRuntime(
    ConvexHullBatchRuntime &&) noexcept = default;
ConvexHullBatchRuntime &
ConvexHullBatchRuntime::operator=(ConvexHullBatchRuntime &&) noexcept =
    default;

namespace {

bool add_int_count(size_t current, size_t addition, size_t &result) {
  if (addition >
      static_cast<size_t>(std::numeric_limits<int>::max()) - current) {
    return false;
  }
  result = current + addition;
  return true;
}

struct RequestRange {
  AssignmentRequest *request;
  size_t candidate_offset;
};

void run_assignment_wave(std::vector<AssignmentRequest> &requests,
                         size_t begin, size_t end,
                         ConvexHullBatchRuntime::Impl &runtime) {
  const size_t job_count = end - begin;
  size_t vertex_count = 0;
  size_t face_count = 0;
  size_t candidate_count = 0;
  for (size_t index = begin; index < end; ++index) {
    const AssignmentRequest &request = requests[index];
    if (!request.state->input.device_mesh &&
        !add_int_count(vertex_count,
                       request.state->input.mesh->vertices.size(),
                       vertex_count)) {
      throw std::overflow_error("Packed hull vertices exceed limits");
    }
    if (!add_int_count(face_count, request.faces.size(), face_count) ||
        !add_int_count(candidate_count, request.candidates.size(),
                       candidate_count)) {
      throw std::overflow_error("Packed hull assignments exceed limits");
    }
  }
  if (job_count >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Packed hull job count exceeds limits");
  }
  if (!candidate_count)
    return;

  runtime.ensure_stream();
  runtime.vertices.ensure(vertex_count * sizeof(double3),
                          "cudaMalloc hull vertices");
  runtime.faces.ensure(face_count * sizeof(double4),
                       "cudaMalloc hull faces");
  runtime.jobs.ensure(job_count * sizeof(PackedAssignmentJob),
                      "cudaMalloc hull jobs");
  runtime.candidate_jobs.ensure(candidate_count * sizeof(int),
                                "cudaMalloc hull candidate jobs");
  runtime.candidate_vertices.ensure(candidate_count * sizeof(int),
                                    "cudaMalloc hull candidates");
  runtime.assignments.ensure(candidate_count * sizeof(int),
                             "cudaMalloc hull assignments");
  runtime.distances.ensure(candidate_count * sizeof(double),
                           "cudaMalloc hull distances");
  runtime.host_vertices.ensure(vertex_count * sizeof(double3),
                               "cudaMallocHost hull vertices");
  runtime.host_faces.ensure(face_count * sizeof(double4),
                            "cudaMallocHost hull faces");
  runtime.host_jobs.ensure(job_count * sizeof(PackedAssignmentJob),
                           "cudaMallocHost hull jobs");
  runtime.host_candidate_jobs.ensure(candidate_count * sizeof(int),
                                     "cudaMallocHost hull candidate jobs");
  runtime.host_candidate_vertices.ensure(
      candidate_count * sizeof(int), "cudaMallocHost hull candidates");
  runtime.host_assignments.ensure(candidate_count * sizeof(int),
                                  "cudaMallocHost hull assignments");
  runtime.host_distances.ensure(candidate_count * sizeof(double),
                                "cudaMallocHost hull distances");

  double3 *host_vertices = runtime.host_vertices.as<double3>();
  double4 *host_faces = runtime.host_faces.as<double4>();
  PackedAssignmentJob *host_jobs =
      runtime.host_jobs.as<PackedAssignmentJob>();
  int *host_candidate_jobs = runtime.host_candidate_jobs.as<int>();
  int *host_candidates = runtime.host_candidate_vertices.as<int>();
  std::vector<RequestRange> ranges;
  ranges.reserve(job_count);
  size_t vertex_offset = 0;
  size_t face_offset = 0;
  size_t candidate_offset = 0;
  for (size_t index = begin; index < end; ++index) {
    AssignmentRequest &request = requests[index];
    HullState &state = *request.state;
    const Mesh &mesh = *state.input.mesh;
    const double3 *vertices = nullptr;
    if (state.input.device_mesh) {
      const DeviceMeshView view =
          device_mesh_view(*state.input.device_mesh);
      if (view.vertex_count != mesh.vertices.size()) {
        throw std::invalid_argument(
            "Hull device mesh vertex count does not match");
      }
      wait_for_device_mesh(*state.input.device_mesh, runtime.stream);
      vertices = reinterpret_cast<const double3 *>(view.vertices);
    } else {
      vertices = runtime.vertices.as<double3>() + vertex_offset;
      for (const Vec3D &vertex : mesh.vertices) {
        host_vertices[vertex_offset++] =
            make_double3(vertex[0], vertex[1], vertex[2]);
      }
    }
    for (int face_index : request.faces)
      host_faces[face_offset++] = state.faces[face_index].plane;
    const size_t local_job = index - begin;
    host_jobs[local_job] = {
        vertices,
        runtime.faces.as<double4>() + face_offset - request.faces.size(),
        static_cast<int>(mesh.vertices.size()),
        static_cast<int>(request.faces.size()),
        state.epsilon};
    for (int vertex : request.candidates) {
      host_candidate_jobs[candidate_offset] =
          static_cast<int>(local_job);
      host_candidates[candidate_offset++] = vertex;
    }
    ranges.push_back(
        {&request, candidate_offset - request.candidates.size()});
  }

  if (vertex_count) {
    check_cuda(cudaMemcpyAsync(runtime.vertices.as<double3>(), host_vertices,
                               vertex_count * sizeof(double3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy hull vertices");
  }
  check_cuda(cudaMemcpyAsync(runtime.faces.as<double4>(), host_faces,
                             face_count * sizeof(double4),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy hull faces");
  check_cuda(cudaMemcpyAsync(runtime.jobs.as<PackedAssignmentJob>(),
                             host_jobs,
                             job_count * sizeof(PackedAssignmentJob),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy hull jobs");
  check_cuda(cudaMemcpyAsync(runtime.candidate_jobs.as<int>(),
                             host_candidate_jobs,
                             candidate_count * sizeof(int),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy hull candidate jobs");
  check_cuda(cudaMemcpyAsync(runtime.candidate_vertices.as<int>(),
                             host_candidates,
                             candidate_count * sizeof(int),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy hull candidates");
  constexpr int block_size = 256;
  const int blocks =
      (static_cast<int>(candidate_count) + block_size - 1) / block_size;
  assign_points_kernel<<<blocks, block_size, 0, runtime.stream>>>(
      runtime.jobs.as<PackedAssignmentJob>(),
      runtime.candidate_jobs.as<int>(),
      runtime.candidate_vertices.as<int>(), runtime.assignments.as<int>(),
      runtime.distances.as<double>(), static_cast<int>(candidate_count));
  check_cuda(cudaGetLastError(), "launch hull point assignment");
  check_cuda(cudaMemcpyAsync(runtime.host_assignments.as<int>(),
                             runtime.assignments.as<int>(),
                             candidate_count * sizeof(int),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy hull assignments");
  check_cuda(cudaMemcpyAsync(runtime.host_distances.as<double>(),
                             runtime.distances.as<double>(),
                             candidate_count * sizeof(double),
                             cudaMemcpyDeviceToHost, runtime.stream),
             "copy hull distances");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "cudaStreamSynchronize hull assignments");

  const int *assignments = runtime.host_assignments.as<int>();
  const double *distances = runtime.host_distances.as<double>();
  for (const RequestRange &range : ranges) {
    scatter_assignments(*range.request,
                        assignments + range.candidate_offset,
                        distances + range.candidate_offset);
  }
}

void classify_requests(std::vector<AssignmentRequest> &requests,
                       ConvexHullBatchRuntime::Impl &runtime,
                       size_t max_batch_size, double memory_fraction) {
  size_t begin = 0;
  while (begin < requests.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo hull");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);
    size_t end = begin;
    size_t vertices = 0;
    size_t faces = 0;
    size_t candidates = 0;
    size_t jobs = 0;
    while (end < requests.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const AssignmentRequest &request = requests[end];
      size_t next_vertices = vertices;
      size_t next_faces = faces;
      size_t next_candidates = candidates;
      if ((!request.state->input.device_mesh &&
           !add_int_count(vertices,
                          request.state->input.mesh->vertices.size(),
                          next_vertices)) ||
          !add_int_count(faces, request.faces.size(), next_faces) ||
          !add_int_count(candidates, request.candidates.size(),
                         next_candidates)) {
        break;
      }
      const size_t growth =
          runtime.growth(next_vertices, next_faces, next_candidates,
                         jobs + 1);
      if (end > begin && growth > budget)
        break;
      vertices = next_vertices;
      faces = next_faces;
      candidates = next_candidates;
      ++jobs;
      ++end;
    }
    if (end == begin)
      ++end;
    run_assignment_wave(requests, begin, end, runtime);
    begin = end;
  }
}

void validate_inputs(const std::vector<ConvexHullBatchInput> &inputs) {
  for (const ConvexHullBatchInput &input : inputs) {
    if (!input.mesh || !input.hull)
      throw std::invalid_argument("Hull input contains a null pointer");
    if (input.mesh->vertices.size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("Hull mesh exceeds indexing limits");
    }
    if (input.device_mesh &&
        device_mesh_view(*input.device_mesh).vertex_count !=
            input.mesh->vertices.size()) {
      throw std::invalid_argument(
          "Hull device mesh vertex count does not match");
    }
  }
}

bool gpu_hull_topology_enabled() {
  // The exact device path currently contends with later GPU stages enough to
  // regress end-to-end throughput, so retain the faster host default.
  const char *enabled = std::getenv("VISACD_ENABLE_GPU_HULL_TOPOLOGY");
  if (!enabled || !*enabled || std::string(enabled) == "0")
    return false;
  const char *disabled = std::getenv("VISACD_DISABLE_GPU_HULL_TOPOLOGY");
  return !disabled || !*disabled || std::string(disabled) == "0";
}

bool topology_capacities(size_t vertex_count, size_t &face_capacity,
                         size_t &horizon_edge_capacity,
                         size_t &output_capacity) {
  constexpr size_t padding = 16;
  const size_t int_limit =
      static_cast<size_t>(std::numeric_limits<int>::max());
  if (vertex_count > (int_limit - padding) / 16)
    return false;
  face_capacity = vertex_count * 4 + padding;
  output_capacity = vertex_count * 2 + padding;
  horizon_edge_capacity = face_capacity * 3;
  return true;
}

struct TopologyOutputRange {
  size_t input_index;
  size_t output_offset;
  size_t output_capacity;
};

void run_topology_wave(
    const std::vector<ConvexHullBatchInput> &inputs, size_t begin,
    size_t end, ConvexHullBatchRuntime::Impl &runtime,
    std::vector<unsigned char> &success) {
  const auto wave_start = std::chrono::steady_clock::now();
  const size_t job_count = end - begin;
  size_t packed_vertex_count = 0;
  size_t vertex_count = 0;
  size_t face_count = 0;
  size_t horizon_edge_count = 0;
  size_t output_count = 0;
  for (size_t index = begin; index < end; ++index) {
    const size_t count = inputs[index].mesh->vertices.size();
    size_t face_capacity = 0;
    size_t horizon_edge_capacity = 0;
    size_t output_capacity = 0;
    if (!topology_capacities(count, face_capacity, horizon_edge_capacity,
                             output_capacity))
      throw std::overflow_error("GPU hull topology capacity overflow");
    vertex_count = checked_add(vertex_count, count,
                               "Packed topology vertices overflow");
    face_count = checked_add(face_count, face_capacity,
                             "Packed topology faces overflow");
    horizon_edge_count = checked_add(
        horizon_edge_count, horizon_edge_capacity,
        "Packed topology horizon edges overflow");
    output_count = checked_add(output_count, output_capacity,
                               "Packed topology outputs overflow");
    if (!inputs[index].device_mesh) {
      packed_vertex_count = checked_add(
          packed_vertex_count, count,
          "Packed topology upload vertices overflow");
    }
  }

  runtime.ensure_stream();
  runtime.vertices.ensure(packed_vertex_count * sizeof(double3),
                          "cudaMalloc topology vertices");
  runtime.topology_faces.ensure(face_count * sizeof(DeviceTopologyFace),
                                "cudaMalloc topology faces");
  runtime.topology_jobs.ensure(job_count * sizeof(PackedTopologyJob),
                               "cudaMalloc topology jobs");
  runtime.topology_assignments.ensure(vertex_count * sizeof(int),
                                      "cudaMalloc topology assignments");
  runtime.topology_distances.ensure(vertex_count * sizeof(double),
                                    "cudaMalloc topology distances");
  runtime.topology_candidates.ensure(
      vertex_count * sizeof(unsigned char),
      "cudaMalloc topology candidate flags");
  runtime.topology_visible.ensure(face_count * sizeof(unsigned char),
                                  "cudaMalloc topology visibility");
  runtime.topology_pending_faces.ensure(
      face_count * sizeof(int), "cudaMalloc topology pending faces");
  runtime.topology_horizon_edges.ensure(
      horizon_edge_count * sizeof(int),
      "cudaMalloc topology horizon edges");
  runtime.topology_output_triangles.ensure(
      output_count * sizeof(int3), "cudaMalloc topology output triangles");
  runtime.topology_output_counts.ensure(job_count * sizeof(int),
                                        "cudaMalloc topology output counts");
  runtime.topology_status.ensure(job_count * sizeof(int),
                                 "cudaMalloc topology status");
  runtime.host_vertices.ensure(packed_vertex_count * sizeof(double3),
                               "cudaMallocHost topology vertices");
  runtime.host_topology_jobs.ensure(
      job_count * sizeof(PackedTopologyJob),
      "cudaMallocHost topology jobs");
  runtime.host_topology_output_triangles.ensure(
      output_count * sizeof(int3),
      "cudaMallocHost topology output triangles");
  runtime.host_topology_output_counts.ensure(
      job_count * sizeof(int), "cudaMallocHost topology output counts");
  runtime.host_topology_status.ensure(
      job_count * sizeof(int), "cudaMallocHost topology status");

  double3 *host_vertices = runtime.host_vertices.as<double3>();
  PackedTopologyJob *host_jobs =
      runtime.host_topology_jobs.as<PackedTopologyJob>();
  std::vector<TopologyOutputRange> ranges;
  ranges.reserve(job_count);
  size_t packed_vertex_offset = 0;
  size_t vertex_offset = 0;
  size_t face_offset = 0;
  size_t horizon_edge_offset = 0;
  size_t output_offset = 0;
  for (size_t local = 0; local < job_count; ++local) {
    const size_t input_index = begin + local;
    const ConvexHullBatchInput &input = inputs[input_index];
    const size_t count = input.mesh->vertices.size();
    size_t face_capacity = 0;
    size_t horizon_edge_capacity = 0;
    size_t output_capacity = 0;
    topology_capacities(count, face_capacity, horizon_edge_capacity,
                        output_capacity);
    const double3 *vertices = nullptr;
    if (input.device_mesh) {
      const DeviceMeshView view = device_mesh_view(*input.device_mesh);
      wait_for_device_mesh(*input.device_mesh, runtime.stream);
      vertices = reinterpret_cast<const double3 *>(view.vertices);
    } else {
      vertices = runtime.vertices.as<double3>() + packed_vertex_offset;
      for (const Vec3D &vertex : input.mesh->vertices) {
        host_vertices[packed_vertex_offset++] =
            make_double3(vertex[0], vertex[1], vertex[2]);
      }
    }
    host_jobs[local] = {
        vertices,
        runtime.topology_faces.as<DeviceTopologyFace>() + face_offset,
        runtime.topology_assignments.as<int>() + vertex_offset,
        runtime.topology_distances.as<double>() + vertex_offset,
        runtime.topology_candidates.as<unsigned char>() + vertex_offset,
        runtime.topology_visible.as<unsigned char>() + face_offset,
        runtime.topology_pending_faces.as<int>() + face_offset,
        runtime.topology_horizon_edges.as<int>() + horizon_edge_offset,
        runtime.topology_output_triangles.as<int3>() + output_offset,
        runtime.topology_output_counts.as<int>() + local,
        runtime.topology_status.as<int>() + local,
        static_cast<int>(count), static_cast<int>(face_capacity),
        static_cast<int>(output_capacity)};
    ranges.push_back({input_index, output_offset, output_capacity});
    vertex_offset += count;
    face_offset += face_capacity;
    horizon_edge_offset += horizon_edge_capacity;
    output_offset += output_capacity;
  }

  if (packed_vertex_count) {
    check_cuda(cudaMemcpyAsync(runtime.vertices.as<double3>(), host_vertices,
                               packed_vertex_count * sizeof(double3),
                               cudaMemcpyHostToDevice, runtime.stream),
               "copy topology vertices");
  }
  check_cuda(cudaMemcpyAsync(runtime.topology_jobs.as<PackedTopologyJob>(),
                             host_jobs,
                             job_count * sizeof(PackedTopologyJob),
                             cudaMemcpyHostToDevice, runtime.stream),
             "copy topology jobs");
  const auto launch_start = std::chrono::steady_clock::now();
  build_hull_topology_kernel<<<static_cast<unsigned int>(job_count), 512, 0,
                               runtime.stream>>>(
      runtime.topology_jobs.as<PackedTopologyJob>(),
      static_cast<int>(job_count));
  check_cuda(cudaGetLastError(), "launch hull topology construction");
  check_cuda(cudaMemcpyAsync(
                 runtime.host_topology_output_counts.as<int>(),
                 runtime.topology_output_counts.as<int>(),
                 job_count * sizeof(int), cudaMemcpyDeviceToHost,
                 runtime.stream),
             "copy topology output counts");
  check_cuda(cudaMemcpyAsync(runtime.host_topology_status.as<int>(),
                             runtime.topology_status.as<int>(),
                             job_count * sizeof(int), cudaMemcpyDeviceToHost,
                             runtime.stream),
             "copy topology status");
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "synchronize hull topology counts");
  const auto counts_done = std::chrono::steady_clock::now();

  const int *host_counts =
      runtime.host_topology_output_counts.as<int>();
  const int *host_status = runtime.host_topology_status.as<int>();
  int3 *host_output =
      runtime.host_topology_output_triangles.as<int3>();
  for (size_t local = 0; local < job_count; ++local) {
    if (host_status[local] <= 0 || host_counts[local] < 4 ||
        static_cast<size_t>(host_counts[local]) >
            ranges[local].output_capacity) {
      continue;
    }
    check_cuda(cudaMemcpyAsync(
                   host_output + ranges[local].output_offset,
                   runtime.topology_output_triangles.as<int3>() +
                       ranges[local].output_offset,
                   static_cast<size_t>(host_counts[local]) * sizeof(int3),
                   cudaMemcpyDeviceToHost, runtime.stream),
               "copy topology output triangles");
  }
  check_cuda(cudaStreamSynchronize(runtime.stream),
             "synchronize hull topology outputs");
  if (config.batch_logging &&
      std::getenv("VISACD_HULL_TOPOLOGY_DIAGNOSTICS")) {
    const auto milliseconds = [](auto duration) {
      return std::chrono::duration<double, std::milli>(duration).count();
    };
    std::fprintf(
        stderr,
        "[visacd hull topology wave] jobs=%zu setup_ms=%.3f "
        "kernel_ms=%.3f output_ms=%.3f\n",
        job_count, milliseconds(launch_start - wave_start),
        milliseconds(counts_done - launch_start),
        milliseconds(std::chrono::steady_clock::now() - counts_done));
  }

  for (size_t local = 0; local < job_count; ++local) {
    if (host_status[local] <= 0 || host_counts[local] < 4)
      continue;
    const ConvexHullBatchInput &input = inputs[ranges[local].input_index];
    std::unordered_map<uint64_t, int> edge_counts;
    bool valid = true;
    for (int face = 0; face < host_counts[local]; ++face) {
      const int3 triangle =
          host_output[ranges[local].output_offset + face];
      const int vertices[3] = {triangle.x, triangle.y, triangle.z};
      for (int edge = 0; edge < 3; ++edge)
        ++edge_counts[edge_key(vertices[edge], vertices[(edge + 1) % 3])];
    }
    for (const auto &edge : edge_counts) {
      if (edge.second != 2) {
        valid = false;
        break;
      }
    }
    if (!valid)
      continue;
    Mesh &hull = *input.hull;
    hull.clear();
    hull.cut_verts.clear();
    hull.is_new.clear();
    hull.intersecting_edges.clear();
    hull.vertices = input.mesh->vertices;
    hull.triangles.reserve(static_cast<size_t>(host_counts[local]));
    for (int face = 0; face < host_counts[local]; ++face) {
      const int3 triangle =
          host_output[ranges[local].output_offset + face];
      hull.triangles.push_back({triangle.x, triangle.y, triangle.z});
    }
    if (input.fix_normals)
      cvx_fix_normals(hull);
    success[ranges[local].input_index] = 1;
    if (config.batch_logging &&
        std::getenv("VISACD_HULL_TOPOLOGY_DIAGNOSTICS")) {
      std::fprintf(stderr,
                   "[visacd hull topology job] input=%zu faces=%d "
                   "active=%d\n",
                   ranges[local].input_index, host_status[local],
                   host_counts[local]);
    }
  }
}

std::vector<unsigned char> compute_gpu_topology(
    const std::vector<ConvexHullBatchInput> &inputs,
    ConvexHullBatchRuntime::Impl &runtime, size_t max_batch_size,
    double memory_fraction) {
  std::vector<unsigned char> success(inputs.size(), 0);
  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo hull topology");
    const size_t budget = static_cast<size_t>(
        static_cast<double>(free_bytes) * memory_fraction);
    size_t end = begin;
    size_t packed_vertices = 0;
    size_t vertices = 0;
    size_t faces = 0;
    size_t horizon_edges = 0;
    size_t outputs = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const size_t count = inputs[end].mesh->vertices.size();
      size_t face_capacity = 0;
      size_t horizon_edge_capacity = 0;
      size_t output_capacity = 0;
      if (!topology_capacities(count, face_capacity, horizon_edge_capacity,
                               output_capacity))
        break;
      const size_t next_packed_vertices =
          inputs[end].device_mesh
              ? packed_vertices
              : checked_add(packed_vertices, count,
                            "Topology wave upload vertices overflow");
      const size_t next_vertices = checked_add(
          vertices, count, "Topology wave vertices overflow");
      const size_t next_faces = checked_add(
          faces, face_capacity, "Topology wave faces overflow");
      const size_t next_horizon_edges = checked_add(
          horizon_edges, horizon_edge_capacity,
          "Topology wave horizon edges overflow");
      const size_t next_outputs = checked_add(
          outputs, output_capacity, "Topology wave outputs overflow");
      const size_t growth = runtime.topology_growth(
          next_packed_vertices, next_vertices, next_faces,
          next_horizon_edges, next_outputs, end - begin + 1);
      if (end > begin && growth > budget)
        break;
      packed_vertices = next_packed_vertices;
      vertices = next_vertices;
      faces = next_faces;
      horizon_edges = next_horizon_edges;
      outputs = next_outputs;
      ++end;
    }
    if (end == begin) {
      ++begin;
      continue;
    }
    run_topology_wave(inputs, begin, end, runtime, success);
    begin = end;
  }
  return success;
}

} // namespace

void compute_convex_hulls_batch(
    const std::vector<ConvexHullBatchInput> &inputs,
    ConvexHullBatchRuntime &runtime, size_t max_batch_size,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument("Hull memory fraction must be in (0, 1]");
  }
  validate_inputs(inputs);
  std::vector<ConvexHullBatchInput> fallback_inputs;
  const std::vector<ConvexHullBatchInput> *host_inputs = &inputs;
  if (gpu_hull_topology_enabled()) {
    const std::vector<unsigned char> success = compute_gpu_topology(
        inputs, *runtime.impl_, max_batch_size, memory_fraction);
    if (config.batch_logging &&
        std::getenv("VISACD_HULL_TOPOLOGY_DIAGNOSTICS")) {
      const size_t completed = static_cast<size_t>(
          std::count(success.begin(), success.end(), 1));
      std::fprintf(stderr,
                   "[visacd hull topology] gpu=%zu fallback=%zu\n",
                   completed, success.size() - completed);
    }
    for (size_t index = 0; index < inputs.size(); ++index) {
      if (!success[index])
        fallback_inputs.push_back(inputs[index]);
    }
    if (fallback_inputs.empty())
      return;
    host_inputs = &fallback_inputs;
  }
  std::vector<HullState> states;
  states.reserve(host_inputs->size());
  for (const ConvexHullBatchInput &input : *host_inputs) {
    HullState state;
    state.input = input;
    states.push_back(std::move(state));
  }

  std::vector<AssignmentRequest> requests;
  requests.reserve(states.size());
  for (HullState &state : states) {
    AssignmentRequest request;
    if (!initialize_hull(state, request)) {
      state.failed = true;
      continue;
    }
    if (!request.candidates.empty())
      requests.push_back(std::move(request));
  }
  classify_requests(requests, *runtime.impl_, max_batch_size,
                    memory_fraction);

  while (true) {
    requests.clear();
    for (HullState &state : states) {
      if (state.failed)
        continue;
      AssignmentRequest request;
      if (prepare_expansion(state, request))
        requests.push_back(std::move(request));
    }
    if (requests.empty())
      break;
    classify_requests(requests, *runtime.impl_, max_batch_size,
                      memory_fraction);
  }

  requests.clear();
  for (HullState &state : states) {
    if (state.failed)
      continue;
    AssignmentRequest request;
    request.state = &state;
    request.validation = true;
    request.candidates.resize(state.input.mesh->vertices.size());
    for (size_t vertex = 0; vertex < request.candidates.size(); ++vertex)
      request.candidates[vertex] = static_cast<int>(vertex);
    for (size_t face = 0; face < state.faces.size(); ++face) {
      if (state.faces[face].active)
        request.faces.push_back(static_cast<int>(face));
    }
    if (!request.candidates.empty() && !request.faces.empty())
      requests.push_back(std::move(request));
  }
  classify_requests(requests, *runtime.impl_, max_batch_size,
                    memory_fraction);

  for (HullState &state : states) {
    if (state.failed || !assemble_hull(state))
      compute_fallback(state.input);
  }
}

} // namespace neural_acd
