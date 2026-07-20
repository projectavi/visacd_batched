#pragma once

#include <core.hpp>
#include <vector>

namespace neural_acd {


struct ProcessResult {
  MeshList parts;
  double concavity;
  int num_parts;
};

ProcessResult process(Mesh mesh, double concavity, int num_parts);

// Decompose a complete batch with shared parameters. Results retain input
// order; independent GPU stages overlap on the current CUDA device.
std::vector<ProcessResult> process_batch(MeshList meshes, double concavity,
                                         int num_parts);

// Release reusable CUDA/OptiX state owned by the calling thread and process.
// No other VisACD decomposition may be active.
void release_gpu_resources();
double compute_final_concavity(MeshList &parts, MeshList &hulls);


} // namespace neural_acd
