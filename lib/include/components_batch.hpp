#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

class DeviceMesh;

struct ComponentBatchInput {
  const Mesh *mesh = nullptr;
  std::vector<int> *labels = nullptr;
  MeshList *components = nullptr;
  const DeviceMesh *projected_edge_source = nullptr;
  const std::vector<int> *projected_vertex_map = nullptr;
  // Optional exact output-vertex to source-vertex maps, one per returned
  // component in the same order as components.
  std::vector<std::vector<int>> *component_vertex_sources = nullptr;
};

class ComponentBatchRuntime {
public:
  ComponentBatchRuntime();
  ~ComponentBatchRuntime();

  ComponentBatchRuntime(const ComponentBatchRuntime &) = delete;
  ComponentBatchRuntime &operator=(const ComponentBatchRuntime &) = delete;
  ComponentBatchRuntime(ComponentBatchRuntime &&) noexcept;
  ComponentBatchRuntime &operator=(ComponentBatchRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void label_components_batch(
      const std::vector<ComponentBatchInput> &, ComponentBatchRuntime &,
      size_t, double);
  friend void separate_components_batch(
      const std::vector<ComponentBatchInput> &, ComponentBatchRuntime &,
      size_t, double);
};

void label_components_batch(const std::vector<ComponentBatchInput> &inputs,
                            ComponentBatchRuntime &runtime,
                            size_t max_batch_size = 0,
                            double memory_fraction = 0.7);

// Labels and stably compacts connected components on the GPU. The returned
// meshes preserve the ordering of assemble_disjoint_parts exactly: component
// order follows first triangle occurrence, vertices follow first corner
// occurrence, and triangles and intersecting edges retain source order. When
// projected_edge_source and projected_vertex_map are provided, the source
// mesh's intersecting edges are projected through the 1-based vertex map on
// the GPU before component assignment.
void separate_components_batch(
    const std::vector<ComponentBatchInput> &inputs,
    ComponentBatchRuntime &runtime, size_t max_batch_size = 0,
    double memory_fraction = 0.7);

} // namespace neural_acd
