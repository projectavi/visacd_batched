#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

struct ComponentBatchInput {
  const Mesh *mesh = nullptr;
  std::vector<int> *labels = nullptr;
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
};

void label_components_batch(const std::vector<ComponentBatchInput> &inputs,
                            ComponentBatchRuntime &runtime,
                            size_t max_batch_size = 0,
                            double memory_fraction = 0.7);

} // namespace neural_acd
