#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace neural_acd {

struct SurfaceVoxelRecord;
class DeviceMesh;

struct ManifoldPreprocessMetrics {
  long long copy_input_ns = 0;
  long long marshal_input_ns = 0;
  long long mesh_to_sdf_ns = 0;
  long long sdf_seed_grid_ns = 0;
  long long sdf_trace_ns = 0;
  long long sdf_sign_ns = 0;
  long long sdf_validate_ns = 0;
  long long sdf_cleanup_ns = 0;
  long long sdf_transform_flood_ns = 0;
  long long sdf_expand_ns = 0;
  long long sdf_renormalize_ns = 0;
  long long volume_to_mesh_ns = 0;
  long long marshal_output_ns = 0;
  size_t input_vertices = 0;
  size_t input_triangles = 0;
  size_t active_voxels = 0;
  size_t output_vertices = 0;
  size_t output_triangles = 0;
};

bool cuda_manifold_preprocessing_enabled();

void manifold_preprocess(Mesh &m, double scale = 50.0f,
                         double level_set = 0.55f,
                         ManifoldPreprocessMetrics *metrics = nullptr);

void manifold_preprocess_cpu_reference(
    Mesh &m, double scale, double level_set,
    ManifoldPreprocessMetrics *metrics = nullptr);

bool manifold_preprocess_cuda_candidate(
    Mesh &m, double scale, double level_set,
    std::string *fallback_reason = nullptr,
    ManifoldPreprocessMetrics *metrics = nullptr);

void manifold_preprocess_from_surface_records(
    Mesh &m, const std::vector<SurfaceVoxelRecord> &surface,
    double scale, double level_set,
    ManifoldPreprocessMetrics *metrics = nullptr,
    std::shared_ptr<DeviceMesh> *device_mesh = nullptr,
    double device_memory_fraction = 0.7);

} // namespace neural_acd
