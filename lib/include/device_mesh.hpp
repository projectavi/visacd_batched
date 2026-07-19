#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

struct DeviceMeshView {
  const double *vertices = nullptr;
  const float *float_vertices = nullptr;
  const int *triangles = nullptr;
  const unsigned int *edges = nullptr;
  size_t vertex_count = 0;
  size_t triangle_count = 0;
  size_t edge_count = 0;
  void *ready_event = nullptr;
};

class DeviceMesh {
public:
  DeviceMesh();
  ~DeviceMesh();

  DeviceMesh(const DeviceMesh &) = delete;
  DeviceMesh &operator=(const DeviceMesh &) = delete;
  DeviceMesh(DeviceMesh &&) noexcept;
  DeviceMesh &operator=(DeviceMesh &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend DeviceMeshView device_mesh_view(const DeviceMesh &);
  friend std::shared_ptr<DeviceMesh> try_make_device_mesh_from_quads(
      const float *, size_t, const int *, size_t, double, void *, double);
  friend class DeviceMeshRuntime;
};

class DeviceMeshRuntime {
public:
  DeviceMeshRuntime();
  ~DeviceMeshRuntime();

  DeviceMeshRuntime(const DeviceMeshRuntime &) = delete;
  DeviceMeshRuntime &operator=(const DeviceMeshRuntime &) = delete;
  DeviceMeshRuntime(DeviceMeshRuntime &&) noexcept;
  DeviceMeshRuntime &operator=(DeviceMeshRuntime &&) noexcept;

  std::shared_ptr<DeviceMesh> try_upload(const Mesh &mesh,
                                         double memory_fraction = 0.7);
  bool try_attach_edges(const std::shared_ptr<DeviceMesh> &device_mesh,
                        const Mesh &mesh,
                        double memory_fraction = 0.7);
  std::shared_ptr<DeviceMesh> try_remap(
      const std::shared_ptr<DeviceMesh> &source, const Mesh &mesh,
      const std::vector<int> &source_vertices,
      double memory_fraction = 0.7);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

DeviceMeshView device_mesh_view(const DeviceMesh &mesh);
void wait_for_device_mesh(const DeviceMesh &mesh, void *stream);

// Retains CUDA level-set meshing output as a regular DeviceMesh without a
// device-to-host-to-device round trip. Points are float3 values in scaled
// preprocessing coordinates and quads are int4 values. The resulting mesh
// applies the same scale division, quad triangulation, and winding as the
// exact host output path.
std::shared_ptr<DeviceMesh> try_make_device_mesh_from_quads(
    const float *device_points, size_t point_count,
    const int *device_quads, size_t quad_count, double scale,
    void *producer_stream, double memory_fraction = 0.7);

} // namespace neural_acd
