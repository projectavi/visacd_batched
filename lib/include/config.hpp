#pragma once

#include <string>

namespace neural_acd {
class Config {
public:
  bool return_parts;

  std::string score_mode; // "edge" or "concavity"

  double flat_surface_min_area;
  bool use_flat_surfaces;
  double flat_surface_k;

  bool use_merging;
  std::string part_limit_policy; // "split_budget" or "adjacent_merge"
  int max_batch_size;           // 0 selects memory-aware automatic sizing
  double batch_memory_fraction; // fraction of currently free device memory
  int batch_cpu_threads;        // 0 scales with batch/hardware, capped at 200
  bool batch_logging;           // emit process_batch progress and diagnostics
  bool retain_gpu_resources;   // retain reusable CUDA/OptiX state across calls

  Config() {
    return_parts = false;

    score_mode = "concavity";

    flat_surface_min_area = 0.1;
    use_flat_surfaces = true;
    flat_surface_k = 2.0;
    use_merging = false;
    part_limit_policy = "split_budget";
    max_batch_size = 0;
    batch_memory_fraction = 0.7;
    batch_cpu_threads = 0;
    batch_logging = true;
    retain_gpu_resources = true;
  }
};

inline Config config;

} // namespace neural_acd
