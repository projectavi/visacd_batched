#pragma once

#include <core.hpp>
#include <cstddef>

namespace neural_acd {

struct ManifoldPreprocessMetrics {
  long long copy_input_ns = 0;
  long long marshal_input_ns = 0;
  long long mesh_to_sdf_ns = 0;
  long long volume_to_mesh_ns = 0;
  long long marshal_output_ns = 0;
  size_t input_vertices = 0;
  size_t input_triangles = 0;
  size_t active_voxels = 0;
  size_t output_vertices = 0;
  size_t output_triangles = 0;
};

void manifold_preprocess(Mesh &m, double scale = 50.0f,
                         double level_set = 0.55f,
                         ManifoldPreprocessMetrics *metrics = nullptr);

} // namespace neural_acd
