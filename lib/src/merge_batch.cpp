#include <algorithm>
#include <array>
#include <batch_executor.hpp>
#include <cmath>
#include <convex_hull_batch.hpp>
#include <cost.hpp>
#include <hausdorff_batch.hpp>
#include <limits>
#include <merge_batch.hpp>
#include <merge_cost_batch.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neural_acd {
namespace {

struct BoundingBox {
  Vec3D minimum{};
  Vec3D maximum{};
};

struct MergeState {
  MergeBatchInput input;
  std::vector<double> part_volumes;
  std::vector<double> hull_volumes;
  std::vector<double> pre_costs;
  std::vector<double> cost_matrix;
  std::vector<double> precost_matrix;
  std::vector<BoundingBox> bounds;
  bool finished = false;
  std::vector<unsigned char> adjacency_matrix;
};

struct PairCandidate {
  size_t state_index = 0;
  size_t first = 0;
  size_t second = 0;
  size_t matrix_index = 0;
  bool needs_proximity = false;
  bool within_threshold = false;
  bool bypass_proximity = false;
  Mesh merged;
  Mesh combined_hull;
  double combined_volume = 0.0;
  double hausdorff = 0.0;
  double cost = INF;
  PreparedHausdorffJob hausdorff_job;
};

BoundingBox compute_bounds(const Mesh &mesh) {
  if (mesh.vertices.empty())
    throw std::invalid_argument("Cannot merge an empty convex hull");
  BoundingBox result{mesh.vertices.front(), mesh.vertices.front()};
  for (const Vec3D &vertex : mesh.vertices) {
    for (int axis = 0; axis < 3; ++axis) {
      result.minimum[axis] =
          std::min(result.minimum[axis], vertex[axis]);
      result.maximum[axis] =
          std::max(result.maximum[axis], vertex[axis]);
    }
  }
  return result;
}

bool bounds_can_overlap(const BoundingBox &first,
                        const BoundingBox &second, double threshold) {
  double distance_squared = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    double distance = 0.0;
    if (first.maximum[axis] < second.minimum[axis])
      distance = second.minimum[axis] - first.maximum[axis];
    else if (second.maximum[axis] < first.minimum[axis])
      distance = first.minimum[axis] - second.maximum[axis];
    distance_squared += distance * distance;
  }
  return distance_squared < threshold * threshold;
}

double radius_from_volume_difference(double difference) {
  return std::pow(3.0 * std::abs(difference) / (4.0 * Pi), 1.0 / 3.0);
}

size_t packed_pair_count(size_t count) {
  if (count > 1 &&
      count - 1 > std::numeric_limits<size_t>::max() / count) {
    throw std::overflow_error("Merge pair count overflow");
  }
  return count * (count - 1) / 2;
}

size_t packed_pair_index(size_t first, size_t second) {
  if (first <= second)
    throw std::logic_error("Packed merge pair is not lower triangular");
  return packed_pair_count(first) + second;
}

size_t packed_pair_index_any_order(size_t first, size_t second) {
  if (first == second)
    throw std::logic_error("Packed merge pair contains one part");
  return packed_pair_index(std::max(first, second),
                           std::min(first, second));
}

Mesh merge_meshes(const Mesh &first, const Mesh &second) {
  if (first.vertices.size() >
      static_cast<size_t>(std::numeric_limits<int>::max()) -
          second.vertices.size()) {
    throw std::overflow_error("Merged hull vertices exceed indexing limits");
  }
  Mesh result;
  result.vertices.reserve(first.vertices.size() + second.vertices.size());
  result.vertices.insert(result.vertices.end(), first.vertices.begin(),
                         first.vertices.end());
  result.vertices.insert(result.vertices.end(), second.vertices.begin(),
                         second.vertices.end());
  result.triangles.reserve(first.triangles.size() +
                           second.triangles.size());
  result.triangles.insert(result.triangles.end(), first.triangles.begin(),
                          first.triangles.end());
  const int offset = static_cast<int>(first.vertices.size());
  for (const auto &triangle : second.triangles) {
    result.triangles.push_back({triangle[0] + offset,
                                triangle[1] + offset,
                                triangle[2] + offset});
  }
  const bool track_interfaces =
      !first.triangle_interfaces.empty() ||
      !second.triangle_interfaces.empty();
  if (track_interfaces) {
    result.triangle_interfaces.reserve(result.triangles.size());
    if (first.triangle_interfaces.empty()) {
      result.triangle_interfaces.insert(
          result.triangle_interfaces.end(), first.triangles.size(), 0);
    } else {
      result.triangle_interfaces.insert(
          result.triangle_interfaces.end(),
          first.triangle_interfaces.begin(),
          first.triangle_interfaces.end());
    }
    if (second.triangle_interfaces.empty()) {
      result.triangle_interfaces.insert(
          result.triangle_interfaces.end(), second.triangles.size(), 0);
    } else {
      result.triangle_interfaces.insert(
          result.triangle_interfaces.end(),
          second.triangle_interfaces.begin(),
          second.triangle_interfaces.end());
    }
  }
  return result;
}

bool parts_are_adjacent(const Mesh &first, const Mesh &second) {
  std::vector<int64_t> interfaces;
  for (int64_t token : first.triangle_interfaces) {
    if (token != 0)
      interfaces.push_back(token);
  }
  std::sort(interfaces.begin(), interfaces.end());
  interfaces.erase(std::unique(interfaces.begin(), interfaces.end()),
                   interfaces.end());
  for (int64_t token : second.triangle_interfaces) {
    if (token != 0 &&
        std::binary_search(interfaces.begin(), interfaces.end(), -token)) {
      return true;
    }
  }
  return false;
}

void validate_inputs(const std::vector<MergeBatchInput> &inputs) {
  for (const MergeBatchInput &input : inputs) {
    if (!input.parts || !input.hulls || !input.part_devices ||
        !input.hull_devices || !input.part_hausdorff || !input.engine) {
      throw std::invalid_argument("Merge batch input contains a null pointer");
    }
    const size_t count = input.hulls->size();
    if (input.parts->size() != count ||
        input.part_devices->size() != count ||
        input.hull_devices->size() != count ||
        input.part_hausdorff->size() != count) {
      throw std::invalid_argument(
          "Merge batch input arrays have different sizes");
    }
    if (input.target_part_count == 0 && !input.use_threshold_merging) {
      throw std::invalid_argument(
          "Merge request has neither a target nor threshold merging");
    }
    for (const Mesh &part : *input.parts) {
      if (!part.triangle_interfaces.empty() &&
          part.triangle_interfaces.size() != part.triangles.size()) {
        throw std::invalid_argument(
            "Merge part interface metadata has the wrong size");
      }
    }
    if (!std::isfinite(input.current_concavity) ||
        input.current_concavity < 0.0 ||
        !std::isfinite(input.threshold) || input.threshold < 0.0) {
      throw std::invalid_argument(
          "Merge thresholds must be finite and non-negative");
    }
  }
}

template <typename Function>
void parallel_finish(BatchExecutor *executor, size_t work_size,
                     Function function) {
  if (executor && executor->thread_count() > 1 && work_size > 1) {
    executor->parallel_for_priority(work_size, std::move(function));
  } else {
    for (size_t index = 0; index < work_size; ++index)
      function(index);
  }
}

} // namespace

struct MergeBatchRuntime::Impl {
  MergeCostBatchRuntime costs;
  ConvexHullBatchRuntime hulls;
  HausdorffRuntime hausdorff;
  DeviceMeshRuntime device_meshes;
};

MergeBatchRuntime::MergeBatchRuntime() : impl_(std::make_unique<Impl>()) {}
MergeBatchRuntime::~MergeBatchRuntime() = default;
MergeBatchRuntime::MergeBatchRuntime(MergeBatchRuntime &&) noexcept =
    default;
MergeBatchRuntime &
MergeBatchRuntime::operator=(MergeBatchRuntime &&) noexcept = default;

void merge_convex_hulls_batch(
    const std::vector<MergeBatchInput> &inputs,
    MergeBatchRuntime &runtime, size_t max_batch_size,
    double memory_fraction, BatchExecutor *executor) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0) {
    throw std::invalid_argument(
        "Merge memory fraction must be in (0, 1]");
  }
  validate_inputs(inputs);

  std::vector<MergeState> states;
  states.reserve(inputs.size());
  for (const MergeBatchInput &input : inputs) {
    MergeState state;
    state.input = input;
    const size_t count = input.hulls->size();
    state.part_volumes.resize(count);
    state.hull_volumes.resize(count);
    state.pre_costs.resize(count);
    state.bounds.reserve(count);
    for (const Mesh &hull : *input.hulls)
      state.bounds.push_back(compute_bounds(hull));
    state.finished =
        count < 2 ||
        (!input.use_threshold_merging &&
         (input.target_part_count == 0 || count <= input.target_part_count));
    states.push_back(std::move(state));
  }

  std::vector<MeshVolumeBatchInput> initial_volume_inputs;
  for (MergeState &state : states) {
    const size_t count = state.input.hulls->size();
    initial_volume_inputs.reserve(initial_volume_inputs.size() + count * 2);
    for (size_t index = 0; index < count; ++index) {
      initial_volume_inputs.push_back(
          {&(*state.input.parts)[index],
           (*state.input.part_devices)[index].get(),
           &state.part_volumes[index]});
      initial_volume_inputs.push_back(
          {&(*state.input.hulls)[index],
           (*state.input.hull_devices)[index].get(),
           &state.hull_volumes[index]});
    }
  }
  evaluate_mesh_volumes_batch(initial_volume_inputs, runtime.impl_->costs,
                              max_batch_size, memory_fraction);
  for (MergeState &state : states) {
    for (size_t index = 0; index < state.pre_costs.size(); ++index) {
      const double relative_volume = radius_from_volume_difference(
          state.part_volumes[index] - state.hull_volumes[index]);
      state.pre_costs[index] =
          std::max(relative_volume * 0.3,
                   (*state.input.part_hausdorff)[index]);
    }
    const size_t count = state.input.hulls->size();
    const size_t matrix_size = packed_pair_count(count);
    state.cost_matrix.assign(matrix_size, INF);
    state.precost_matrix.resize(matrix_size);
    state.adjacency_matrix.assign(matrix_size, 0);
    for (size_t first = 1; first < count; ++first) {
      for (size_t second = 0; second < first; ++second) {
        const size_t pair_index = packed_pair_index(first, second);
        state.precost_matrix[pair_index] =
            std::max(state.pre_costs[first], state.pre_costs[second]);
        state.adjacency_matrix[pair_index] =
            parts_are_adjacent((*state.input.parts)[first],
                               (*state.input.parts)[second])
                ? 1
                : 0;
      }
    }
  }

  const auto build_candidate_hulls =
      [&](std::vector<PairCandidate *> &active_candidates) {
        parallel_finish(executor, active_candidates.size(), [&](size_t index) {
          PairCandidate &candidate = *active_candidates[index];
          MergeState &state = states[candidate.state_index];
          candidate.merged =
              merge_meshes((*state.input.hulls)[candidate.first],
                           (*state.input.hulls)[candidate.second]);
        });

        std::vector<ConvexHullBatchInput> hull_inputs;
        hull_inputs.reserve(active_candidates.size());
        for (PairCandidate *candidate : active_candidates) {
          hull_inputs.push_back(
              {&candidate->merged, nullptr,
               &candidate->combined_hull, true});
        }
        const size_t cpu_threads =
            executor ? executor->thread_count() : 1;
        const size_t gpu_crossover =
            std::max<size_t>(32, cpu_threads * 4);
        if (active_candidates.size() >= gpu_crossover) {
          compute_convex_hulls_batch(hull_inputs, runtime.impl_->hulls,
                                     max_batch_size, memory_fraction);
        } else {
          parallel_finish(
              executor, active_candidates.size(), [&](size_t index) {
                PairCandidate &candidate = *active_candidates[index];
                candidate.merged.compute_ch(candidate.combined_hull, true);
              });
        }

        std::vector<MeshVolumeBatchInput> volume_inputs;
        volume_inputs.reserve(active_candidates.size());
        for (PairCandidate *candidate : active_candidates) {
          volume_inputs.push_back(
              {&candidate->combined_hull, nullptr,
               &candidate->combined_volume});
        }
        evaluate_mesh_volumes_batch(
            volume_inputs, runtime.impl_->costs, max_batch_size,
            memory_fraction);
      };

  const auto evaluate_candidates =
      [&](std::vector<PairCandidate> &candidates) {
    if (candidates.empty())
      return;
    std::vector<MeshProximityBatchInput> proximity_inputs;
    for (PairCandidate &candidate : candidates) {
      MergeState &state = states[candidate.state_index];
      if (candidate.bypass_proximity) {
        candidate.within_threshold = true;
        continue;
      }
      if (!state.input.use_threshold_merging)
        continue;
      candidate.needs_proximity =
          bounds_can_overlap(state.bounds[candidate.first],
                             state.bounds[candidate.second],
                             state.input.threshold);
      if (!candidate.needs_proximity)
        continue;
      proximity_inputs.push_back(
          {&(*state.input.hulls)[candidate.first],
           (*state.input.hull_devices)[candidate.first].get(),
           &(*state.input.hulls)[candidate.second],
           (*state.input.hull_devices)[candidate.second].get(),
           state.input.threshold, &candidate.within_threshold});
    }
    evaluate_mesh_proximity_batch(proximity_inputs, runtime.impl_->costs,
                                  max_batch_size, memory_fraction);

    std::vector<PairCandidate *> active_candidates;
    for (PairCandidate &candidate : candidates) {
      if (candidate.within_threshold)
        active_candidates.push_back(&candidate);
    }
    build_candidate_hulls(active_candidates);

    std::vector<PreparedHausdorffJob *> hausdorff_jobs;
    for (PairCandidate *candidate : active_candidates) {
      MergeState &state = states[candidate->state_index];
      const Mesh &first = (*state.input.hulls)[candidate->first];
      const Mesh &second = (*state.input.hulls)[candidate->second];
      if (candidate->combined_hull.vertices.size() ==
          first.vertices.size() + second.vertices.size()) {
        candidate->hausdorff = 0.0;
        continue;
      }
      candidate->hausdorff_job = prepare_merge_hausdorff_job(
          (*state.input.hulls)[candidate->first],
          (*state.input.hulls)[candidate->second], candidate->merged,
          candidate->combined_hull, 12000, *state.input.engine);
      hausdorff_jobs.push_back(&candidate->hausdorff_job);
    }
    evaluate_hausdorff_batch(hausdorff_jobs, runtime.impl_->hausdorff,
                             max_batch_size, memory_fraction);

    for (PairCandidate *candidate : active_candidates) {
      MergeState &state = states[candidate->state_index];
      if (candidate->hausdorff_job.valid)
        candidate->hausdorff = candidate->hausdorff_job.result;
      else if (candidate->combined_hull.vertices.size() !=
               (*state.input.hulls)[candidate->first].vertices.size() +
                   (*state.input.hulls)[candidate->second].vertices.size()) {
        candidate->hausdorff = INF;
      }
      const double relative_volume = radius_from_volume_difference(
          state.hull_volumes[candidate->first] +
          state.hull_volumes[candidate->second] -
          candidate->combined_volume);
      candidate->cost =
          std::max(relative_volume * 0.3, candidate->hausdorff);
    }
  };

  size_t initial_pair_count = 0;
  for (const MergeState &state : states) {
    const size_t addition = state.cost_matrix.size();
    if (addition > std::numeric_limits<size_t>::max() -
                       initial_pair_count) {
      throw std::overflow_error("Merge pair count overflow");
    }
    initial_pair_count += addition;
  }
  std::vector<PairCandidate> initial_candidates;
  initial_candidates.reserve(initial_pair_count);
  for (size_t state_index = 0; state_index < states.size(); ++state_index) {
    const size_t count = states[state_index].input.hulls->size();
    for (size_t first = 1; first < count; ++first) {
      for (size_t second = 0; second < first; ++second) {
        PairCandidate candidate;
        candidate.state_index = state_index;
        candidate.first = first;
        candidate.second = second;
        candidate.matrix_index = packed_pair_index(first, second);
        candidate.bypass_proximity =
            states[state_index].input.target_part_count > 0 &&
            states[state_index].adjacency_matrix[candidate.matrix_index] != 0;
        initial_candidates.push_back(std::move(candidate));
      }
    }
  }
  evaluate_candidates(initial_candidates);
  for (const PairCandidate &candidate : initial_candidates) {
    states[candidate.state_index].cost_matrix[candidate.matrix_index] =
        candidate.cost;
  }
  initial_candidates.clear();
  initial_candidates.shrink_to_fit();

  while (true) {
    std::vector<PairCandidate> accepted;
    accepted.reserve(states.size());
    for (size_t state_index = 0; state_index < states.size();
         ++state_index) {
      MergeState &state = states[state_index];
      if (state.finished)
        continue;
      const bool enforcing_limit =
          state.input.target_part_count > 0 &&
          state.input.hulls->size() > state.input.target_part_count;
      if (!enforcing_limit && !state.input.use_threshold_merging) {
        state.finished = true;
        continue;
      }
      while (true) {
        size_t best_index = state.cost_matrix.size();
        double best_cost = INF;
        for (size_t index = 0; index < state.cost_matrix.size(); ++index) {
          if (enforcing_limit && state.adjacency_matrix[index] == 0)
            continue;
          if (best_index == state.cost_matrix.size() ||
              state.cost_matrix[index] < best_cost) {
            best_cost = state.cost_matrix[index];
            best_index = index;
          }
        }
        if (best_index == state.cost_matrix.size()) {
          if (enforcing_limit) {
            throw std::runtime_error(
                "Adjacent part limit is infeasible: no split-provenance "
                "adjacent merge remains");
          }
          state.finished = true;
          break;
        }
        if (!enforcing_limit && best_cost > state.input.threshold) {
          state.finished = true;
          break;
        }
        if (!enforcing_limit &&
            best_cost + state.precost_matrix[best_index] >
            state.input.current_concavity) {
          state.cost_matrix[best_index] = INF;
          continue;
        }

        size_t first = 1;
        while (packed_pair_count(first + 1) <= best_index)
          ++first;
        PairCandidate candidate;
        candidate.state_index = state_index;
        candidate.first = first;
        candidate.second = best_index - packed_pair_count(first);
        candidate.matrix_index = best_index;
        candidate.cost = best_cost;
        accepted.push_back(std::move(candidate));
        break;
      }
    }

    if (accepted.empty())
      break;

    std::vector<PairCandidate *> accepted_pointers;
    accepted_pointers.reserve(accepted.size());
    for (PairCandidate &candidate : accepted)
      accepted_pointers.push_back(&candidate);
    build_candidate_hulls(accepted_pointers);
    for (PairCandidate &candidate : accepted) {
      MergeState &state = states[candidate.state_index];
      MeshList &parts = *state.input.parts;
      MeshList &hulls = *state.input.hulls;
      auto &part_devices = *state.input.part_devices;
      auto &hull_devices = *state.input.hull_devices;
      std::vector<double> &part_hausdorff =
          *state.input.part_hausdorff;
      const size_t first = candidate.first;
      const size_t second = candidate.second;
      const size_t old_count = hulls.size();
      const size_t last = old_count - 1;

      for (size_t other = 0; other < old_count; ++other) {
        if (other == first || other == second)
          continue;
        const size_t second_edge =
            packed_pair_index_any_order(second, other);
        const size_t first_edge =
            packed_pair_index_any_order(first, other);
        state.adjacency_matrix[second_edge] =
            state.adjacency_matrix[second_edge] ||
                    state.adjacency_matrix[first_edge]
                ? 1
                : 0;
      }
      if (first != last) {
        for (size_t other = 0; other < last; ++other) {
          if (other == first)
            continue;
          state.adjacency_matrix[
              packed_pair_index_any_order(first, other)] =
              state.adjacency_matrix[
                  packed_pair_index_any_order(last, other)];
        }
      }
      state.adjacency_matrix.resize(packed_pair_count(last));

      parts[second] = merge_meshes(parts[first], parts[second]);
      part_devices[second] = runtime.impl_->device_meshes.try_upload(
          parts[second], memory_fraction);
      state.part_volumes[second] += state.part_volumes[first];
      state.pre_costs[second] =
          std::max(state.pre_costs[first], state.pre_costs[second]) +
          candidate.cost;
      part_hausdorff[second] = std::max(
          {part_hausdorff[first], part_hausdorff[second],
           candidate.hausdorff});

      hulls[second] = std::move(candidate.combined_hull);
      hull_devices[second] = runtime.impl_->device_meshes.try_upload(
          hulls[second], memory_fraction);
      state.hull_volumes[second] = candidate.combined_volume;
      state.bounds[second] = compute_bounds(hulls[second]);

      if (first != last) {
        std::swap(parts[first], parts[last]);
        std::swap(hulls[first], hulls[last]);
        std::swap(part_devices[first], part_devices[last]);
        std::swap(hull_devices[first], hull_devices[last]);
        std::swap(part_hausdorff[first], part_hausdorff[last]);
        std::swap(state.part_volumes[first],
                  state.part_volumes[last]);
        std::swap(state.pre_costs[first], state.pre_costs[last]);
        std::swap(state.hull_volumes[first],
                  state.hull_volumes[last]);
        std::swap(state.bounds[first], state.bounds[last]);
      }
      parts.pop_back();
      hulls.pop_back();
      part_devices.pop_back();
      hull_devices.pop_back();
      part_hausdorff.pop_back();
      state.part_volumes.pop_back();
      state.pre_costs.pop_back();
      state.hull_volumes.pop_back();
      state.bounds.pop_back();
    }

    size_t update_count = 0;
    for (const PairCandidate &candidate : accepted) {
      const size_t count =
          states[candidate.state_index].input.hulls->size();
      if (count > 0) {
        if (count - 1 > std::numeric_limits<size_t>::max() -
                            update_count) {
          throw std::overflow_error("Merge update count overflow");
        }
        update_count += count - 1;
      }
    }
    std::vector<PairCandidate> updates;
    updates.reserve(update_count);
    for (const PairCandidate &merge : accepted) {
      MergeState &state = states[merge.state_index];
      const size_t count = state.input.hulls->size();
      for (size_t other = 0; other < count; ++other) {
        if (other == merge.second)
          continue;
        PairCandidate candidate;
        candidate.state_index = merge.state_index;
        candidate.first = std::max(other, merge.second);
        candidate.second = std::min(other, merge.second);
        candidate.matrix_index =
            packed_pair_index(candidate.first, candidate.second);
        candidate.bypass_proximity =
            state.input.target_part_count > 0 &&
            state.adjacency_matrix[candidate.matrix_index] != 0;
        updates.push_back(std::move(candidate));
      }
    }
    evaluate_candidates(updates);
    for (const PairCandidate &candidate : updates) {
      states[candidate.state_index].cost_matrix[candidate.matrix_index] =
          candidate.cost;
    }

    for (const PairCandidate &merge : accepted) {
      MergeState &state = states[merge.state_index];
      const size_t first = merge.first;
      const size_t second = merge.second;
      const size_t count = state.input.hulls->size();

      for (size_t other = 0; other < second; ++other) {
        const size_t index = packed_pair_index(second, other);
        state.precost_matrix[index] =
            std::max(state.precost_matrix[second] + merge.cost,
                     state.precost_matrix[other]);
      }
      for (size_t other = second + 1; other < count; ++other) {
        const size_t index = packed_pair_index(other, second);
        state.precost_matrix[index] =
            std::max(state.precost_matrix[second] + merge.cost,
                     state.precost_matrix[other]);
      }

      const size_t erase_index = packed_pair_count(count);
      const size_t old_last = count;
      if (first < count) {
        for (size_t other = 0; other < first; ++other) {
          if (other == second)
            continue;
          const size_t destination = packed_pair_index(first, other);
          const size_t source = packed_pair_index(old_last, other);
          state.cost_matrix[destination] = state.cost_matrix[source];
          state.precost_matrix[destination] =
              state.precost_matrix[source];
        }
        for (size_t other = first + 1; other < count; ++other) {
          const size_t destination = packed_pair_index(other, first);
          const size_t source = packed_pair_index(old_last, other);
          state.cost_matrix[destination] = state.cost_matrix[source];
          state.precost_matrix[destination] =
              state.precost_matrix[source];
        }
      }
      state.cost_matrix.resize(erase_index);
      state.precost_matrix.resize(erase_index);
      if (state.adjacency_matrix.size() != erase_index)
        throw std::logic_error("Merge adjacency matrix is out of sync");
      state.finished =
          count < 2 ||
          (!state.input.use_threshold_merging &&
           (state.input.target_part_count == 0 ||
            count <= state.input.target_part_count));
    }
  }
}

} // namespace neural_acd
