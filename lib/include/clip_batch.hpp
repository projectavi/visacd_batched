#pragma once

#include <core.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace neural_acd {

class DeviceMesh;

struct ClipTriangleData {
  short sides[3] = {0, 0, 0};
  unsigned short intersection_mask = 0;
  double intersections[9] = {0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0};
};

struct ClipBatchInput {
  const DeviceMesh *device_mesh = nullptr;
  Plane plane;
  std::vector<ClipTriangleData> *output = nullptr;
};

class ClipBatchRuntime {
public:
  ClipBatchRuntime();
  ~ClipBatchRuntime();

  ClipBatchRuntime(const ClipBatchRuntime &) = delete;
  ClipBatchRuntime &operator=(const ClipBatchRuntime &) = delete;
  ClipBatchRuntime(ClipBatchRuntime &&) noexcept;
  ClipBatchRuntime &operator=(ClipBatchRuntime &&) noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;

  friend void prepare_clip_batch(const std::vector<ClipBatchInput> &,
                                 ClipBatchRuntime &, size_t, double);
};

void prepare_clip_batch(const std::vector<ClipBatchInput> &inputs,
                        ClipBatchRuntime &runtime,
                        size_t max_batch_size = 0,
                        double memory_fraction = 0.7);

} // namespace neural_acd
