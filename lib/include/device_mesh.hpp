#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>

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

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

DeviceMeshView device_mesh_view(const DeviceMesh &mesh);
void wait_for_device_mesh(const DeviceMesh &mesh, void *stream);

} // namespace neural_acd
