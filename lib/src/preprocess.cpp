#include <algorithm>
#include <chrono>
#include <core.hpp>
#include <cstdio>
#include <iostream>
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

void sdf_manifold(Mesh &input, Mesh &output, double scale, double level_set,
                  ManifoldPreprocessMetrics *metrics) {
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
  DoubleGrid::Ptr sgrid = tools::meshToSignedDistanceField<DoubleGrid>(
      *xform, points, tris, quads, level_set * scale + 1.0, 3.0);
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
}

void manifold_preprocess(Mesh &m, double scale, double level_set,
                         ManifoldPreprocessMetrics *metrics) {
  if (metrics)
    *metrics = ManifoldPreprocessMetrics{};
  const auto copy_start = PreprocessClock::now();
  Mesh tmp = m;
  m.clear();
  if (metrics)
    metrics->copy_input_ns = elapsed_ns(copy_start);
  sdf_manifold(tmp, m, scale, level_set, metrics);
}
} // namespace neural_acd
