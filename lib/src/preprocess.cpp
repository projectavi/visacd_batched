#include <algorithm>
#include <chrono>
#include <core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/util/Util.h>
#include <preprocess.hpp>
#include <preprocess_cuda.hpp>
#include <string>
#include <vector>

using namespace openvdb;
using namespace std;

namespace neural_acd {

namespace {

using PreprocessClock = chrono::steady_clock;

long long elapsed_ns(PreprocessClock::time_point start) {
  return chrono::duration_cast<chrono::nanoseconds>(
             PreprocessClock::now() - start)
      .count();
}

bool environment_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && string(value) != "0";
}

// This is the post-voxelization portion of OpenVDB 8.2's MeshToVolume
// pipeline, specialized for the exact surface records produced on CUDA.
DoubleGrid::Ptr signed_distance_field_from_surface(
    const vector<Vec3s> &points, const vector<Vec3I> &triangles,
    const math::Transform &transform,
    const vector<SurfaceVoxelRecord> &surface, float exterior_band_width,
    float interior_band_width) {
  using GridType = DoubleGrid;
  using TreeType = GridType::TreeType;
  using LeafNodeType = TreeType::LeafNodeType;
  using ValueType = GridType::ValueType;
  using IntGridType = GridType::ValueConverter<Int32>::Type;
  using IntTreeType = IntGridType::TreeType;
  using BoolTreeType = TreeType::ValueConverter<bool>::Type;
  using Adapter = tools::QuadAndTriangleDataAdapter<Vec3s, Vec3I>;

  GridType::Ptr distance_grid(
      new GridType(numeric_limits<ValueType>::max()));
  distance_grid->setTransform(transform.copy());
  distance_grid->setGridClass(GRID_LEVEL_SET);

  IntGridType index_grid(Int32(util::INVALID_IDX));
  index_grid.setTransform(transform.copy());
  TreeType &distance_tree = distance_grid->tree();
  IntTreeType &index_tree = index_grid.tree();
  tree::ValueAccessor<TreeType> distance_accessor(distance_tree);
  tree::ValueAccessor<IntTreeType> index_accessor(index_tree);
  for (const SurfaceVoxelRecord &voxel : surface) {
    const Coord coordinate(voxel.x, voxel.y, voxel.z);
    distance_accessor.setValue(coordinate, voxel.squared_distance);
    index_accessor.setValue(coordinate, voxel.triangle_index);
  }

  const ValueType voxel_size = ValueType(transform.voxelSize()[0]);
  ValueType exterior_width = ValueType(exterior_band_width) * voxel_size;
  ValueType interior_width = ValueType(interior_band_width);
  if (interior_width < numeric_limits<ValueType>::max())
    interior_width *= voxel_size;
  Adapter mesh(points, triangles);

  tools::traceExteriorBoundaries(distance_tree);
  vector<LeafNodeType *> nodes;
  nodes.reserve(distance_tree.leafCount());
  distance_tree.getNodes(nodes);
  const tbb::blocked_range<size_t> node_range(0, nodes.size());
  using SignOperation =
      tools::mesh_to_volume_internal::ComputeIntersectingVoxelSign<
          TreeType, Adapter>;
  tbb::parallel_for(node_range,
                    SignOperation(nodes, distance_tree, index_tree, mesh));
  tbb::parallel_for(
      node_range,
      tools::mesh_to_volume_internal::ValidateIntersectingVoxels<TreeType>(
          distance_tree, nodes));
  tbb::parallel_for(
      node_range,
      tools::mesh_to_volume_internal::RemoveSelfIntersectingSurface<TreeType>(
          nodes, distance_tree, index_tree));
  tools::pruneInactive(distance_tree, true);
  tools::pruneInactive(index_tree, true);

  if (distance_tree.activeVoxelCount() == 0) {
    distance_tree.clear();
    distance_tree.root().setBackground(exterior_width, false);
    return distance_grid;
  }

  nodes.clear();
  nodes.reserve(distance_tree.leafCount());
  distance_tree.getNodes(nodes);
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, nodes.size()),
      tools::mesh_to_volume_internal::TransformValues<TreeType>(nodes,
                                                                 voxel_size,
                                                                 false));
  distance_tree.root().setBackground(exterior_width, false);
  tools::signedFloodFillWithValues(distance_tree, exterior_width,
                                   -interior_width);

  const ValueType minimum_band_width = voxel_size * ValueType(2.0);
  if (interior_width > minimum_band_width ||
      exterior_width > minimum_band_width) {
    BoolTreeType mask_tree(false);
    nodes.clear();
    nodes.reserve(distance_tree.leafCount());
    distance_tree.getNodes(nodes);
    tools::mesh_to_volume_internal::ConstructVoxelMask<TreeType> mask_builder(
        mask_tree, distance_tree, nodes);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()),
                         mask_builder);

    unsigned maximum_iterations = numeric_limits<unsigned>::max();
    const double estimated =
        2.0 * ceil((max(interior_width, exterior_width) -
                    minimum_band_width) /
                   voxel_size);
    if (estimated < double(maximum_iterations))
      maximum_iterations = unsigned(estimated);

    vector<BoolTreeType::LeafNodeType *> mask_nodes;
    unsigned iteration = 0;
    while (true) {
      const size_t mask_node_count = mask_tree.leafCount();
      if (mask_node_count == 0)
        break;
      mask_nodes.clear();
      mask_nodes.reserve(mask_node_count);
      mask_tree.getNodes(mask_nodes);
      const tbb::blocked_range<size_t> range(0, mask_nodes.size());
      tbb::parallel_for(
          range,
          tools::mesh_to_volume_internal::DiffLeafNodeMask<TreeType>(
              distance_tree, mask_nodes));
      tools::mesh_to_volume_internal::expandNarrowband(
          distance_tree, index_tree, mask_tree, mask_nodes, mesh,
          exterior_width, interior_width, voxel_size);
      if (++iteration >= maximum_iterations)
        break;
    }
  }

  index_grid.clear();
  nodes.clear();
  nodes.reserve(distance_tree.leafCount());
  distance_tree.getNodes(nodes);
  unique_ptr<ValueType[]> buffer(
      new ValueType[LeafNodeType::SIZE * nodes.size()]);
  const ValueType offset = ValueType(0.8 * voxel_size);
  const tbb::blocked_range<size_t> final_node_range(0, nodes.size());
  tbb::parallel_for(
      final_node_range,
      tools::mesh_to_volume_internal::OffsetValues<TreeType>(nodes,
                                                               -offset));
  tbb::parallel_for(
      final_node_range,
      tools::mesh_to_volume_internal::Renormalize<TreeType>(
          distance_tree, nodes, buffer.get(), voxel_size));
  tbb::parallel_for(
      final_node_range,
      tools::mesh_to_volume_internal::MinCombine<TreeType>(nodes,
                                                            buffer.get()));
  tbb::parallel_for(
      final_node_range,
      tools::mesh_to_volume_internal::OffsetValues<TreeType>(
          nodes,
          offset -
              tools::mesh_to_volume_internal::Tolerance<ValueType>::epsilon()));

  if (min(interior_width, exterior_width) <
      voxel_size * ValueType(4.0)) {
    tbb::parallel_for(
        final_node_range,
        tools::mesh_to_volume_internal::InactivateValues<TreeType>(
            nodes, exterior_width, interior_width));
    tools::pruneLevelSet(distance_tree, exterior_width, -interior_width);
  }
  return distance_grid;
}

bool meshes_match_exactly(const Mesh &first, const Mesh &second) {
  if (first.vertices.size() != second.vertices.size() ||
      first.triangles.size() != second.triangles.size())
    return false;
  for (size_t index = 0; index < first.vertices.size(); ++index) {
    for (int axis = 0; axis < 3; ++axis) {
      if (memcmp(&first.vertices[index][axis],
                 &second.vertices[index][axis], sizeof(double)) != 0)
        return false;
    }
  }
  return first.triangles == second.triangles;
}

} // namespace

vector<SurfaceVoxelRecord>
reference_surface_voxelization(const Mesh &input, double scale) {
  vector<Vec3s> points;
  vector<Vec3I> triangles;
  points.reserve(input.vertices.size());
  triangles.reserve(input.triangles.size());
  for (const Vec3D &vertex : input.vertices) {
    points.push_back({static_cast<float>(vertex[0] * scale),
                      static_cast<float>(vertex[1] * scale),
                      static_cast<float>(vertex[2] * scale)});
  }
  for (const array<int, 3> &triangle : input.triangles) {
    triangles.push_back({static_cast<unsigned int>(triangle[0]),
                         static_cast<unsigned int>(triangle[1]),
                         static_cast<unsigned int>(triangle[2])});
  }
  if (points.empty() || triangles.empty())
    return {};

  using TreeType = DoubleGrid::TreeType;
  using IntTreeType = TreeType::ValueConverter<Int32>::Type;
  using VoxelizationData =
      tools::mesh_to_volume_internal::VoxelizationData<TreeType>;
  using DataTable =
      tbb::enumerable_thread_specific<typename VoxelizationData::Ptr>;
  using Adapter = tools::QuadAndTriangleDataAdapter<Vec3s, Vec3I>;
  using Voxelizer =
      tools::mesh_to_volume_internal::VoxelizePolygons<TreeType, Adapter>;

  Adapter adapter(points, triangles);
  DataTable data;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, adapter.polygonCount()),
                    Voxelizer(data, adapter));

  TreeType distance_tree(numeric_limits<double>::max());
  IntTreeType index_tree(Int32(util::INVALID_IDX));
  for (auto iterator = data.begin(); iterator != data.end(); ++iterator) {
    VoxelizationData &item = **iterator;
    tools::mesh_to_volume_internal::combineData(
        distance_tree, index_tree, item.distTree, item.indexTree);
  }

  tree::ValueAccessor<const IntTreeType> index_accessor(index_tree);
  vector<SurfaceVoxelRecord> records;
  records.reserve(distance_tree.activeVoxelCount());
  for (auto iterator = distance_tree.cbeginValueOn(); iterator; ++iterator) {
    const Coord coordinate = iterator.getCoord();
    records.push_back({coordinate.x(), coordinate.y(), coordinate.z(),
                       *iterator, index_accessor.getValue(coordinate)});
  }
  sort(records.begin(), records.end(), [](const auto &first,
                                          const auto &second) {
    if (first.x != second.x)
      return first.x < second.x;
    if (first.y != second.y)
      return first.y < second.y;
    return first.z < second.z;
  });
  return records;
}

bool sdf_manifold(Mesh &input, Mesh &output, double scale, double level_set,
                  bool use_cuda, string *fallback_reason,
                  ManifoldPreprocessMetrics *metrics,
                  const vector<SurfaceVoxelRecord> *provided_surface =
                      nullptr) {
  const auto marshal_start = PreprocessClock::now();
  vector<Vec3s> points;
  vector<Vec3I> tris;
  vector<Vec4I> quads;

  points.reserve(input.vertices.size());
  tris.reserve(input.triangles.size());

  for (unsigned int i = 0; i < input.vertices.size(); ++i) {
    points.push_back({(float)(input.vertices[i][0] * scale),
                      (float)(input.vertices[i][1] * scale),
                      (float)(input.vertices[i][2] * scale)});
  }
  for (unsigned int i = 0; i < input.triangles.size(); ++i) {
    tris.push_back({(unsigned int)input.triangles[i][0],
                    (unsigned int)input.triangles[i][1],
                    (unsigned int)input.triangles[i][2]});
  }

  math::Transform::Ptr xform = math::Transform::createLinearTransform();
  tools::QuadAndTriangleDataAdapter<Vec3s, Vec3I> mesh(points, tris);
  if (metrics) {
    metrics->input_vertices = input.vertices.size();
    metrics->input_triangles = input.triangles.size();
    metrics->marshal_input_ns = elapsed_ns(marshal_start);
  }

  const auto sdf_start = PreprocessClock::now();
  DoubleGrid::Ptr sgrid;
  if (provided_surface) {
    sgrid = signed_distance_field_from_surface(
        points, tris, *xform, *provided_surface,
        static_cast<float>(level_set * scale + 1.0), 3.0f);
  } else if (use_cuda) {
    try {
      static thread_local ManifoldCudaRuntime runtime;
      SurfaceVoxelizationResult surface =
          runtime.voxelize_surface(input, scale);
      if (!surface.supported) {
        if (fallback_reason)
          *fallback_reason = surface.fallback_reason;
        return false;
      }
      sgrid = signed_distance_field_from_surface(
          points, tris, *xform, surface.records,
          static_cast<float>(level_set * scale + 1.0), 3.0f);
    } catch (const exception &error) {
      if (fallback_reason)
        *fallback_reason = error.what();
      return false;
    }
  } else {
    sgrid = tools::meshToSignedDistanceField<DoubleGrid>(
        *xform, points, tris, quads, level_set * scale + 1.0, 3.0);
  }
  if (metrics) {
    metrics->mesh_to_sdf_ns = elapsed_ns(sdf_start);
    metrics->active_voxels = sgrid->activeVoxelCount();
  }

  vector<Vec3s> newPoints;
  vector<Vec3I> newTriangles;
  vector<Vec4I> newQuads;
  const auto meshing_start = PreprocessClock::now();
  tools::volumeToMesh(*sgrid, newPoints, newTriangles, newQuads, level_set*scale);
  if (metrics)
    metrics->volume_to_mesh_ns = elapsed_ns(meshing_start);

  const auto output_start = PreprocessClock::now();
  output.clear();
  output.vertices.reserve(newPoints.size());
  output.triangles.reserve(newTriangles.size() + newQuads.size() * 2);
  for (unsigned int i = 0; i < newPoints.size(); ++i) {
    output.vertices.push_back({newPoints[i][0] / scale, newPoints[i][1] / scale,
                               newPoints[i][2] / scale});
  }
  for (unsigned int i = 0; i < newTriangles.size(); ++i) {
    output.triangles.push_back({(int)newTriangles[i][0],
                                (int)newTriangles[i][2],
                                (int)newTriangles[i][1]});
  }
  for (unsigned int i = 0; i < newQuads.size(); ++i) {
    output.triangles.push_back(
        {(int)newQuads[i][0], (int)newQuads[i][2], (int)newQuads[i][1]});
    output.triangles.push_back(
        {(int)newQuads[i][0], (int)newQuads[i][3], (int)newQuads[i][2]});
  }
  if (metrics) {
    metrics->marshal_output_ns = elapsed_ns(output_start);
    metrics->output_vertices = output.vertices.size();
    metrics->output_triangles = output.triangles.size();
  }
  return true;
}

void manifold_preprocess_cpu_reference(Mesh &m, double scale,
                                       double level_set,
                                       ManifoldPreprocessMetrics *metrics) {
  if (metrics)
    *metrics = ManifoldPreprocessMetrics{};
  const auto copy_start = PreprocessClock::now();
  Mesh tmp = m;
  if (metrics)
    metrics->copy_input_ns = elapsed_ns(copy_start);
  Mesh output;
  sdf_manifold(tmp, output, scale, level_set, false, nullptr, metrics);
  m = std::move(output);
}

bool manifold_preprocess_cuda_candidate(
    Mesh &m, double scale, double level_set, string *fallback_reason,
    ManifoldPreprocessMetrics *metrics) {
  if (metrics)
    *metrics = ManifoldPreprocessMetrics{};
  const auto copy_start = PreprocessClock::now();
  Mesh tmp = m;
  if (metrics)
    metrics->copy_input_ns = elapsed_ns(copy_start);
  Mesh output;
  if (!sdf_manifold(tmp, output, scale, level_set, true,
                    fallback_reason, metrics))
    return false;
  m = std::move(output);
  return true;
}

void manifold_preprocess_from_surface_records(
    Mesh &m, const vector<SurfaceVoxelRecord> &surface, double scale,
    double level_set, ManifoldPreprocessMetrics *metrics) {
  if (metrics)
    *metrics = ManifoldPreprocessMetrics{};
  const auto copy_start = PreprocessClock::now();
  Mesh tmp = m;
  if (metrics)
    metrics->copy_input_ns = elapsed_ns(copy_start);
  Mesh output;
  sdf_manifold(tmp, output, scale, level_set, false, nullptr, metrics,
               &surface);
  m = std::move(output);
}

void manifold_preprocess(Mesh &m, double scale, double level_set,
                         ManifoldPreprocessMetrics *metrics) {
  if (!environment_enabled("VISACD_ENABLE_CUDA_PREPROCESS")) {
    manifold_preprocess_cpu_reference(m, scale, level_set, metrics);
    return;
  }

  if (environment_enabled("VISACD_VERIFY_CUDA_PREPROCESS")) {
    Mesh reference = m;
    Mesh candidate = m;
    ManifoldPreprocessMetrics candidate_metrics;
    string fallback_reason;
    if (manifold_preprocess_cuda_candidate(
            candidate, scale, level_set, &fallback_reason,
            metrics ? &candidate_metrics : nullptr)) {
      ManifoldPreprocessMetrics reference_metrics;
      manifold_preprocess_cpu_reference(
          reference, scale, level_set,
          metrics ? &reference_metrics : nullptr);
      if (meshes_match_exactly(reference, candidate)) {
        m = std::move(candidate);
        if (metrics)
          *metrics = candidate_metrics;
        return;
      }
      if (environment_enabled("VISACD_PREPROCESS_TRACE"))
        cerr << "preprocess_cuda_fallback reason=full_output_mismatch\n";
      m = std::move(reference);
      if (metrics)
        *metrics = reference_metrics;
      return;
    }
    if (environment_enabled("VISACD_PREPROCESS_TRACE"))
      cerr << "preprocess_cuda_fallback reason=" << fallback_reason << '\n';
    ManifoldPreprocessMetrics reference_metrics;
    manifold_preprocess_cpu_reference(
        reference, scale, level_set,
        metrics ? &reference_metrics : nullptr);
    m = std::move(reference);
    if (metrics)
      *metrics = reference_metrics;
    return;
  }

  string fallback_reason;
  if (manifold_preprocess_cuda_candidate(m, scale, level_set,
                                         &fallback_reason, metrics))
    return;
  if (environment_enabled("VISACD_PREPROCESS_TRACE"))
    cerr << "preprocess_cuda_fallback reason=" << fallback_reason << '\n';
  manifold_preprocess_cpu_reference(m, scale, level_set, metrics);
}
} // namespace neural_acd
