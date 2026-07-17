#pragma once

#include "core.hpp"
#include <cstdint>
#include <vector>

namespace neural_acd {

struct Surface {
  std::vector<int> triangle_ids;
  double area = 0.0;
  Plane plane;
};

std::vector<Surface> extract_surfaces(const Mesh &mesh, double min_area);

std::vector<Surface> assemble_surfaces_from_features(
    const Mesh &mesh, double min_area,
    const std::vector<Vec3D> &normals,
    const std::vector<double> &areas,
    const std::vector<std::uint64_t> &edge_keys,
    const std::vector<std::uint64_t> &sorted_edge_keys,
    const std::vector<int> &sorted_edge_triangles);

} // namespace neural_acd
