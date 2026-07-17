#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <support_surface.hpp>
#include <unordered_map>
#include <vector>

namespace neural_acd {
namespace {

double cosine_sim(const Vec3D &first, const Vec3D &second) {
  return dot(first, second) /
         (std::sqrt(dot(first, first)) * std::sqrt(dot(second, second)));
}

std::vector<Surface> assemble_regions(
    const Mesh &mesh, double min_area, const std::vector<Vec3D> &normals,
    const std::vector<double> &areas,
    const std::vector<std::vector<int>> &adjacency) {
  const size_t triangle_count = mesh.triangles.size();
  if (normals.size() != triangle_count || areas.size() != triangle_count ||
      adjacency.size() != triangle_count) {
    throw std::invalid_argument(
        "Flat-surface features do not match the mesh");
  }

  std::vector<bool> used(triangle_count, false);
  std::vector<Surface> surfaces;
  for (size_t start = 0; start < triangle_count; ++start) {
    if (used[start])
      continue;

    Surface surface;
    surface.triangle_ids.push_back(static_cast<int>(start));
    used[start] = true;
    Vec3D average_normal = normals[start];

    std::queue<int> pending;
    pending.push(static_cast<int>(start));
    while (!pending.empty()) {
      const int triangle = pending.front();
      pending.pop();
      for (int neighbor : adjacency[triangle]) {
        if (used[neighbor])
          continue;
        if (cosine_sim(average_normal, normals[neighbor]) > 0.999) {
          used[neighbor] = true;
          surface.triangle_ids.push_back(neighbor);
          average_normal =
              (average_normal * (surface.triangle_ids.size() - 1) +
               normals[neighbor]) /
              surface.triangle_ids.size();
          pending.push(neighbor);
        }
      }
    }
    surfaces.push_back(std::move(surface));
  }

  std::vector<Surface> filtered;
  for (Surface &surface : surfaces) {
    double area = 0.0;
    for (int triangle : surface.triangle_ids)
      area += areas[triangle];
    surface.area = area;
    if (area > min_area)
      filtered.push_back(surface);
  }

  if (filtered.size() > 40) {
    std::sort(filtered.begin(), filtered.end(),
              [](const Surface &first, const Surface &second) {
                return first.area > second.area;
              });
    filtered.resize(40);
  }

  const auto &triangles = mesh.triangles;
  const auto &vertices = mesh.vertices;
  for (Surface &surface : filtered) {
    Vec3D average_normal = {0.0, 0.0, 0.0};
    Vec3D point_on_surface = {0.0, 0.0, 0.0};
    for (int triangle_id : surface.triangle_ids) {
      const auto &triangle = triangles[triangle_id];
      average_normal =
          average_normal +
          calc_face_normal(vertices[triangle[0]], vertices[triangle[1]],
                           vertices[triangle[2]]);
      point_on_surface =
          point_on_surface +
          (vertices[triangle[0]] + vertices[triangle[1]] +
           vertices[triangle[2]]) /
              3.0;
    }
    average_normal = normalize_vector(average_normal);
    point_on_surface =
        point_on_surface / surface.triangle_ids.size();
    point_on_surface = point_on_surface + average_normal * 2e-2;
    surface.plane =
        Plane(average_normal[0], average_normal[1], average_normal[2],
              -dot(average_normal, point_on_surface));
  }

  std::vector<Surface> unique_surfaces;
  for (const Surface &surface : filtered) {
    bool duplicate = false;
    for (const Surface &existing : unique_surfaces) {
      const Plane &first = surface.plane;
      const Plane &second = existing.plane;
      const double dot_product =
          first.a * second.a + first.b * second.b + first.c * second.c +
          first.d * second.d;
      const double first_magnitude =
          std::sqrt(first.a * first.a + first.b * first.b +
                    first.c * first.c + first.d * first.d);
      const double second_magnitude =
          std::sqrt(second.a * second.a + second.b * second.b +
                    second.c * second.c + second.d * second.d);
      if (first_magnitude > 1e-9 && second_magnitude > 1e-9) {
        const double cosine =
            std::abs(dot_product) / (first_magnitude * second_magnitude);
        if (cosine > 1.0 - 1e-4) {
          duplicate = true;
          break;
        }
      }
    }
    if (!duplicate)
      unique_surfaces.push_back(surface);
  }
  return unique_surfaces;
}

} // namespace

std::vector<Surface> assemble_surfaces_from_features(
    const Mesh &mesh, double min_area,
    const std::vector<Vec3D> &normals,
    const std::vector<double> &areas,
    const std::vector<std::uint64_t> &edge_keys,
    const std::vector<std::uint64_t> &sorted_edge_keys,
    const std::vector<int> &sorted_edge_triangles) {
  const size_t triangle_count = mesh.triangles.size();
  if (normals.size() != triangle_count || areas.size() != triangle_count ||
      edge_keys.size() != triangle_count * 3 ||
      sorted_edge_keys.size() != triangle_count * 3 ||
      sorted_edge_triangles.size() != sorted_edge_keys.size()) {
    throw std::invalid_argument(
        "GPU flat-surface features do not match the mesh");
  }

  struct EdgeRange {
    size_t begin = 0;
    size_t end = 0;
  };
  std::unordered_map<std::uint64_t, EdgeRange> edge_ranges;
  edge_ranges.reserve(triangle_count * 3);
  // Insert in the legacy triangle-edge order before attaching GPU-sorted
  // ranges. This retains the original region-growing neighbor order.
  for (std::uint64_t key : edge_keys)
    edge_ranges.try_emplace(key);

  size_t edge_begin = 0;
  while (edge_begin < sorted_edge_keys.size()) {
    size_t edge_end = edge_begin + 1;
    while (edge_end < sorted_edge_keys.size() &&
           sorted_edge_keys[edge_end] == sorted_edge_keys[edge_begin]) {
      ++edge_end;
    }
    const auto found = edge_ranges.find(sorted_edge_keys[edge_begin]);
    if (found == edge_ranges.end()) {
      throw std::invalid_argument(
          "GPU flat-surface edge sets do not match");
    }
    found->second = {edge_begin, edge_end};
    edge_begin = edge_end;
  }

  std::vector<std::vector<int>> adjacency(triangle_count);
  for (const auto &entry : edge_ranges) {
    const EdgeRange range = entry.second;
    if (range.end <= range.begin) {
      throw std::invalid_argument(
          "GPU flat-surface edge group is missing");
    }
    for (size_t first = range.begin; first < range.end; ++first) {
      const int first_triangle = sorted_edge_triangles[first];
      if (first_triangle < 0 ||
          static_cast<size_t>(first_triangle) >= triangle_count) {
        throw std::invalid_argument(
            "GPU flat-surface edge has an invalid triangle index");
      }
      for (size_t second = range.begin; second < range.end; ++second) {
        const int second_triangle = sorted_edge_triangles[second];
        if (second_triangle < 0 ||
            static_cast<size_t>(second_triangle) >= triangle_count) {
          throw std::invalid_argument(
              "GPU flat-surface edge has an invalid triangle index");
        }
        if (first_triangle != second_triangle)
          adjacency[first_triangle].push_back(second_triangle);
      }
    }
  }
  return assemble_regions(mesh, min_area, normals, areas, adjacency);
}

std::vector<Surface> extract_surfaces(const Mesh &mesh, double min_area) {
  const auto &triangles = mesh.triangles;
  const auto &vertices = mesh.vertices;
  const size_t triangle_count = triangles.size();

  std::vector<Vec3D> normals(triangle_count);
  std::vector<double> areas(triangle_count);
  for (size_t index = 0; index < triangle_count; ++index) {
    const auto &triangle = triangles[index];
    normals[index] =
        calc_face_normal(vertices[triangle[0]], vertices[triangle[1]],
                         vertices[triangle[2]]);
    areas[index] =
        triangle_area(vertices[triangle[0]], vertices[triangle[1]],
                      vertices[triangle[2]]);
  }

  std::unordered_map<std::uint64_t, std::vector<int>> edge_map;
  edge_map.reserve(triangle_count * 3);
  const auto encode = [](int first, int second) {
    if (first > second)
      std::swap(first, second);
    return (static_cast<std::uint64_t>(
                static_cast<unsigned int>(first))
            << 32) |
           static_cast<unsigned int>(second);
  };
  for (size_t index = 0; index < triangle_count; ++index) {
    const auto &triangle = triangles[index];
    edge_map[encode(triangle[0], triangle[1])].push_back(
        static_cast<int>(index));
    edge_map[encode(triangle[1], triangle[2])].push_back(
        static_cast<int>(index));
    edge_map[encode(triangle[2], triangle[0])].push_back(
        static_cast<int>(index));
  }

  std::vector<std::vector<int>> adjacency(triangle_count);
  for (const auto &entry : edge_map) {
    const std::vector<int> &touching = entry.second;
    for (int first : touching) {
      for (int second : touching) {
        if (first != second)
          adjacency[first].push_back(second);
      }
    }
  }
  return assemble_regions(mesh, min_area, normals, areas, adjacency);
}

} // namespace neural_acd
