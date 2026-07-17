#include <algorithm>
#include <array>
#include <atomic>
#include <batch_executor.hpp>
#include <candidate_planes.hpp>
#include <chrono>
#include <clip.hpp>
#include <config.hpp>
#include <condition_variable>
#include <components_batch.hpp>
#include <convex_hull_batch.hpp>
#include <core.hpp>
#include <cost.hpp>
#include <deque>
#include <device_mesh.hpp>
#include <exception>
#include <functional>
#include <hausdorff_batch.hpp>
#include <iomanip>
#include <intersections.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <plane_intersections.hpp>
#include <postprocess.hpp>
#include <preprocess.hpp>
#include <process.hpp>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <support_surface.hpp>
#include <utility>
#include <vector>

using namespace std;

namespace neural_acd {

// Existing decomposition helpers retained in process.cpp.
vector<Plane>
get_candidate_planes(vector<Vec3D> &vertices,
                     vector<pair<unsigned int, unsigned int>> &edges,
                     int num_planes);
vector<Plane>
get_candidate_planes(vector<Vec3D> &vertices,
                     vector<pair<unsigned int, unsigned int>> &edges,
                     int num_planes, RandomEngine &engine);
int get_part_with_highest_score(MeshList &parts);
int get_part_with_highest_concavity(MeshList &parts, double &max_concavity);
int get_part_with_highest_concavity(MeshList &parts, double &max_concavity,
                                    RandomEngine &engine);
namespace {

constexpr size_t kMaxAutomaticCpuThreads = 200;
constexpr size_t kCandidatePlaneCount = 2500;
constexpr size_t kCandidateAttemptMultiplier = 5;

struct PartCache {
  optional<Mesh> hull;
  optional<double> selection_concavity;
  shared_ptr<DeviceMesh> device;
  shared_ptr<DeviceMesh> hull_device;
};

struct BatchState {
  Mesh cage;
  MeshList parts;
  vector<PartCache> part_cache;
  vector<double> original_bbox;
  vector<Plane> flat_surface_planes;
  RandomEngine random_engine;
  int iteration = 0;
  bool finished = false;
  bool finalizing = false;
};

struct SplitWork {
  size_t state_index;
  Mesh part;
  PartCache part_cache;
  vector<Plane> planes;
  size_t flat_surface_offset;
  vector<unsigned int> sampled_edges;
  size_t candidate_attempts = 0;
  vector<float> host_planes;
  vector<float> host_points;
  vector<unsigned int> host_edges;
  vector<float> scores;
  size_t selected_plane = 0;
  vector<ClipTriangleData> prepared_clip;
};

struct PendingPart {
  size_t state_index;
  Mesh part;
  Mesh cage;
};

struct IntersectionWork {
  size_t state_index;
  bool initial = false;
  vector<PendingPart> pending_parts;
  vector<pair<Mesh *, Mesh *>> requests;
};

struct ComponentWork {
  size_t state_index;
  bool initial = false;
  MeshList parts;
  vector<vector<int>> labels;
};

struct FinalizeWork {
  size_t state_index;
  MeshList hulls;
  vector<shared_ptr<DeviceMesh>> hull_devices;
  atomic<size_t> remaining{0};
};

struct HullWork {
  size_t state_index;
  vector<size_t> part_indices;
};

enum class HausdorffPurpose { selection, final_concavity };

struct HausdorffWork {
  HausdorffPurpose purpose;
  size_t state_index;
  vector<size_t> part_indices;
  vector<PreparedHausdorffJob> jobs;
  vector<double> relative_volume_terms;
  shared_ptr<FinalizeWork> finalize_work;
};

enum class GpuWorkKind {
  intersections,
  planes,
  hausdorff,
  components,
  hulls,
  complete
};

struct GpuWork {
  GpuWorkKind kind;
  vector<shared_ptr<IntersectionWork>> intersections;
  vector<shared_ptr<SplitWork>> planes;
  vector<shared_ptr<HausdorffWork>> hausdorff;
  vector<shared_ptr<ComponentWork>> components;
  vector<shared_ptr<HullWork>> hulls;
};

class PipelineCoordinator {
public:
  PipelineCoordinator(BatchExecutor &executor, size_t total_states,
                      size_t gpu_batch_threshold)
      : executor_(executor), total_states_(total_states),
        gpu_batch_threshold_(max<size_t>(1, gpu_batch_threshold)) {}

  void submit_cpu(function<void()> task, bool priority = false) {
    {
      lock_guard<mutex> lock(mutex_);
      if (error_)
        return;
      ++active_cpu_tasks_;
    }

    auto wrapped = [this, task = move(task)]() mutable {
      exception_ptr error;
      bool should_run;
      {
        lock_guard<mutex> lock(mutex_);
        should_run = !error_;
      }
      if (should_run) {
        try {
          task();
        } catch (...) {
          error = current_exception();
        }
      }
      finish_cpu_task(error);
    };

    try {
      if (priority)
        executor_.submit_priority(move(wrapped));
      else
        executor_.submit(move(wrapped));
    } catch (...) {
      finish_cpu_task(current_exception());
    }
  }

  void enqueue_intersections(shared_ptr<IntersectionWork> work) {
    lock_guard<mutex> lock(mutex_);
    if (error_)
      return;
    intersection_request_count_ += work->requests.size();
    intersection_queue_.push_back(move(work));
    condition_.notify_one();
  }

  void enqueue_planes(shared_ptr<SplitWork> work) {
    lock_guard<mutex> lock(mutex_);
    if (error_)
      return;
    plane_queue_.push_back(move(work));
    condition_.notify_one();
  }

  void enqueue_hausdorff(shared_ptr<HausdorffWork> work) {
    lock_guard<mutex> lock(mutex_);
    if (error_)
      return;
    hausdorff_request_count_ += work->jobs.size();
    hausdorff_queue_.push_back(move(work));
    condition_.notify_one();
  }

  void enqueue_components(shared_ptr<ComponentWork> work) {
    lock_guard<mutex> lock(mutex_);
    if (error_)
      return;
    component_request_count_ += work->parts.size();
    component_queue_.push_back(move(work));
    condition_.notify_one();
  }

  void enqueue_hulls(shared_ptr<HullWork> work) {
    lock_guard<mutex> lock(mutex_);
    if (error_)
      return;
    hull_request_count_ += work->part_indices.size();
    hull_queue_.push_back(move(work));
    condition_.notify_one();
  }

  void complete_hull_batch() {
    lock_guard<mutex> lock(mutex_);
    hull_busy_ = false;
    condition_.notify_one();
  }

  void complete_state() {
    lock_guard<mutex> lock(mutex_);
    ++completed_states_;
    condition_.notify_one();
  }

  void set_error(exception_ptr error) {
    lock_guard<mutex> lock(mutex_);
    record_error(error);
  }

  GpuWork wait_for_gpu_work() {
    unique_lock<mutex> lock(mutex_);
    while (true) {
      condition_.wait(lock, [this]() {
        return error_ || completed() || !intersection_queue_.empty() ||
               !plane_queue_.empty() || !hausdorff_queue_.empty() ||
               !component_queue_.empty() ||
               (!hull_busy_ && !hull_queue_.empty());
      });

      if (error_) {
        if (active_cpu_tasks_ != 0) {
          condition_.wait(lock, [this]() { return active_cpu_tasks_ == 0; });
        }
        rethrow_exception(error_);
      }
      if (completed())
        return {GpuWorkKind::complete, {}, {}, {}, {}, {}};

      if (!gpu_batch_ready() && active_cpu_tasks_ != 0) {
        condition_.wait_for(lock, chrono::milliseconds(1), [this]() {
          return error_ || completed() || gpu_batch_ready() ||
                 active_cpu_tasks_ == 0;
        });
        if (error_)
          continue;
        if (completed())
          return {GpuWorkKind::complete, {}, {}, {}, {}, {}};
      }

      const array<bool, 5> available = {!intersection_queue_.empty(),
                                        !plane_queue_.empty(),
                                        !hausdorff_queue_.empty(),
                                        !component_queue_.empty(),
                                        !hull_busy_ &&
                                            !hull_queue_.empty()};
      size_t selected = 0;
      for (; selected < available.size(); ++selected) {
        const size_t candidate = (next_gpu_kind_ + selected) % available.size();
        if (available[candidate]) {
          selected = candidate;
          break;
        }
      }
      next_gpu_kind_ = (selected + 1) % available.size();

      if (selected == 0) {
        GpuWork result{GpuWorkKind::intersections, {}, {}, {}, {}, {}};
        result.intersections.reserve(intersection_queue_.size());
        while (!intersection_queue_.empty()) {
          shared_ptr<IntersectionWork> work =
              move(intersection_queue_.front());
          intersection_queue_.pop_front();
          intersection_request_count_ -= work->requests.size();
          result.intersections.push_back(move(work));
        }
        return result;
      }

      if (selected == 1) {
        GpuWork result{GpuWorkKind::planes, {}, {}, {}, {}, {}};
        result.planes.reserve(plane_queue_.size());
        while (!plane_queue_.empty()) {
          result.planes.push_back(move(plane_queue_.front()));
          plane_queue_.pop_front();
        }
        return result;
      }

      if (selected == 2) {
        GpuWork result{GpuWorkKind::hausdorff, {}, {}, {}, {}, {}};
        result.hausdorff.reserve(hausdorff_queue_.size());
        while (!hausdorff_queue_.empty()) {
          shared_ptr<HausdorffWork> work = move(hausdorff_queue_.front());
          hausdorff_queue_.pop_front();
          hausdorff_request_count_ -= work->jobs.size();
          result.hausdorff.push_back(move(work));
        }
        return result;
      }

      if (selected == 3) {
        GpuWork result{GpuWorkKind::components, {}, {}, {}, {}, {}};
        result.components.reserve(component_queue_.size());
        while (!component_queue_.empty()) {
          shared_ptr<ComponentWork> work = move(component_queue_.front());
          component_queue_.pop_front();
          component_request_count_ -= work->parts.size();
          result.components.push_back(move(work));
        }
        return result;
      }

      GpuWork result{GpuWorkKind::hulls, {}, {}, {}, {}, {}};
      hull_busy_ = true;
      result.hulls.reserve(hull_queue_.size());
      while (!hull_queue_.empty()) {
        shared_ptr<HullWork> work = move(hull_queue_.front());
        hull_queue_.pop_front();
        hull_request_count_ -= work->part_indices.size();
        result.hulls.push_back(move(work));
      }
      return result;
    }
  }

private:
  bool completed() const {
    return completed_states_ == total_states_ && active_cpu_tasks_ == 0;
  }

  bool gpu_batch_ready() const {
    return intersection_request_count_ >= gpu_batch_threshold_ ||
           plane_queue_.size() >= gpu_batch_threshold_ ||
           hausdorff_request_count_ >= gpu_batch_threshold_ ||
           component_request_count_ >= gpu_batch_threshold_ ||
           (!hull_busy_ &&
            hull_request_count_ >= gpu_batch_threshold_);
  }

  void record_error(exception_ptr error) {
    if (!error_)
      error_ = error;
    intersection_queue_.clear();
    plane_queue_.clear();
    hausdorff_queue_.clear();
    component_queue_.clear();
    hull_queue_.clear();
    intersection_request_count_ = 0;
    hausdorff_request_count_ = 0;
    component_request_count_ = 0;
    hull_request_count_ = 0;
    hull_busy_ = false;
    condition_.notify_all();
  }

  void finish_cpu_task(exception_ptr error) {
    lock_guard<mutex> lock(mutex_);
    if (error)
      record_error(error);
    --active_cpu_tasks_;
    condition_.notify_all();
  }

  BatchExecutor &executor_;
  const size_t total_states_;
  const size_t gpu_batch_threshold_;
  deque<shared_ptr<IntersectionWork>> intersection_queue_;
  deque<shared_ptr<SplitWork>> plane_queue_;
  deque<shared_ptr<HausdorffWork>> hausdorff_queue_;
  deque<shared_ptr<ComponentWork>> component_queue_;
  deque<shared_ptr<HullWork>> hull_queue_;
  size_t intersection_request_count_ = 0;
  size_t hausdorff_request_count_ = 0;
  size_t component_request_count_ = 0;
  size_t hull_request_count_ = 0;
  size_t active_cpu_tasks_ = 0;
  size_t completed_states_ = 0;
  size_t next_gpu_kind_ = 0;
  bool hull_busy_ = false;
  mutex mutex_;
  condition_variable condition_;
  exception_ptr error_;
};

void log(size_t mesh_index, const string &message) {
  static mutex log_mutex;
  lock_guard<mutex> lock(log_mutex);
  cout << "[visacd batch " << mesh_index << "] " << message << "\n";
  cout.flush();
}

size_t configured_cpu_threads(size_t work_size) {
  if (work_size == 0)
    return 0;
  size_t requested = static_cast<size_t>(config.batch_cpu_threads);
  if (requested == 0) {
    const unsigned int available = thread::hardware_concurrency();
    const size_t hardware_threads = available == 0 ? 2 : available;
    requested = min(hardware_threads, kMaxAutomaticCpuThreads);
  }
  return max<size_t>(1, min(requested, work_size));
}

void validate_parameters(const MeshList &meshes, double concavity,
                         int num_parts) {
  if (num_parts < 1)
    throw invalid_argument("num_parts must be at least 1");
  if (!isfinite(concavity) || concavity < 0.0)
    throw invalid_argument("concavity must be a finite non-negative value");
  if (config.max_batch_size < 0)
    throw invalid_argument("max_batch_size cannot be negative");
  if (config.batch_cpu_threads < 0)
    throw invalid_argument("batch_cpu_threads cannot be negative");
  if (config.batch_memory_fraction <= 0.0 ||
      config.batch_memory_fraction > 1.0) {
    throw invalid_argument("batch_memory_fraction must be in (0, 1]");
  }

  for (const Mesh &mesh : meshes) {
    if (mesh.vertices.empty() || mesh.triangles.empty())
      throw invalid_argument("Each input mesh must contain triangles");
  }
}

void initialize_state(Mesh mesh, size_t mesh_index, BatchState &state) {
  state.original_bbox = mesh.normalize();
  log(mesh_index, "Preprocessing mesh (" + to_string(mesh.vertices.size()) +
                      " verts)...");

  Mesh original_mesh = mesh.copy();
  manifold_preprocess(mesh, 30, 0.55 / 30);
  if (mesh.vertices.size() > 15000) {
    mesh = original_mesh.copy();
    manifold_preprocess(mesh, 20, 0.55 / 20);
  }
  log(mesh_index,
      "Remeshed to " + to_string(mesh.vertices.size()) + " verts.");

  state.cage = mesh.copy();
  manifold_preprocess(state.cage, 40, 0.03);

  if (config.use_flat_surfaces) {
    vector<Surface> surfaces =
        extract_surfaces(mesh, config.flat_surface_min_area);
    log(mesh_index,
        "Detected " + to_string(surfaces.size()) + " flat surface(s).");
    state.flat_surface_planes.reserve(surfaces.size());
    for (const auto &surface : surfaces)
      state.flat_surface_planes.push_back(surface.plane);
  }

  state.parts.push_back(move(mesh));
}

SplitWork prepare_split_work(size_t state_index, Mesh part,
                             PartCache part_cache, RandomEngine &engine) {
  SplitWork work;
  work.state_index = state_index;
  work.part = move(part);
  work.part_cache = move(part_cache);
  if (work.part.intersecting_edges.size() >
      static_cast<size_t>(numeric_limits<int>::max())) {
    throw overflow_error("Candidate edge count exceeds indexing limits");
  }
  RandomEngine preview_engine = engine;
  uniform_int_distribution<> distribution(
      0, static_cast<int>(work.part.intersecting_edges.size()) - 1);
  work.sampled_edges.resize(kCandidatePlaneCount *
                            kCandidateAttemptMultiplier);
  for (unsigned int &sample : work.sampled_edges)
    sample = static_cast<unsigned int>(distribution(preview_engine));
  return work;
}

void finish_split_work(SplitWork &work,
                       const vector<Plane> &flat_surface_planes,
                       RandomEngine &engine) {
  if (work.candidate_attempts > work.sampled_edges.size())
    throw logic_error("Candidate kernel consumed too many samples");
  uniform_int_distribution<> distribution(
      0, static_cast<int>(work.part.intersecting_edges.size()) - 1);
  for (size_t attempt = 0; attempt < work.candidate_attempts; ++attempt) {
    const unsigned int sampled =
        static_cast<unsigned int>(distribution(engine));
    if (sampled != work.sampled_edges[attempt])
      throw logic_error("Candidate random stream is out of sync");
  }

  work.flat_surface_offset = work.planes.size();
  work.planes.insert(work.planes.end(), flat_surface_planes.begin(),
                     flat_surface_planes.end());

  work.host_planes.resize(work.planes.size() * 4);
  for (size_t i = 0; i < work.planes.size(); ++i) {
    work.host_planes[i * 4] = static_cast<float>(work.planes[i].a);
    work.host_planes[i * 4 + 1] = static_cast<float>(work.planes[i].b);
    work.host_planes[i * 4 + 2] = static_cast<float>(work.planes[i].c);
    work.host_planes[i * 4 + 3] = static_cast<float>(work.planes[i].d);
  }

  if (!work.part_cache.device) {
    work.host_points.resize(work.part.vertices.size() * 3);
    for (size_t i = 0; i < work.part.vertices.size(); ++i) {
      work.host_points[i * 3] =
          static_cast<float>(work.part.vertices[i][0]);
      work.host_points[i * 3 + 1] =
          static_cast<float>(work.part.vertices[i][1]);
      work.host_points[i * 3 + 2] =
          static_cast<float>(work.part.vertices[i][2]);
    }

    work.host_edges.resize(work.part.intersecting_edges.size() * 2);
    for (size_t i = 0; i < work.part.intersecting_edges.size(); ++i) {
      work.host_edges[i * 2] = work.part.intersecting_edges[i].first;
      work.host_edges[i * 2 + 1] =
          work.part.intersecting_edges[i].second;
    }
  }
  work.scores.assign(work.planes.size(), 0.0f);
}

int get_part_with_highest_cached_concavity(const BatchState &state,
                                           double &max_concavity) {
  if (state.parts.size() != state.part_cache.size())
    throw logic_error("Part cache is out of sync with batch state");

  int best_index = -1;
  for (size_t part_index = 0; part_index < state.parts.size(); ++part_index) {
    const PartCache &cache = state.part_cache[part_index];
    if (!cache.selection_concavity)
      throw logic_error("Part concavity was not evaluated before selection");
    if (*cache.selection_concavity > max_concavity) {
      max_concavity = *cache.selection_concavity;
      best_index = static_cast<int>(part_index);
    }
  }
  return best_index;
}

void propagate_existing_edges(
    const Mesh &part, const int *first_map, const int *second_map,
    MeshList &new_parts) {
  for (const auto &edge : part.intersecting_edges) {
    const int first = edge.first;
    const int second = edge.second;
    if (first_map[first] && first_map[second]) {
      new_parts[0].intersecting_edges.emplace_back(first_map[first] - 1,
                                                    first_map[second] - 1);
    }
    if (second_map[first] && second_map[second]) {
      new_parts[1].intersecting_edges.emplace_back(second_map[first] - 1,
                                                    second_map[second] - 1);
    }
  }
}

void append_intersections(
    const vector<pair<Mesh *, Mesh *>> &requests,
    const vector<vector<pair<unsigned int, unsigned int>>> &edges,
    BatchExecutor &executor) {
  executor.parallel_for_priority(requests.size(), [&](size_t i) {
    auto &destination = requests[i].first->intersecting_edges;
    destination.insert(destination.end(), edges[i].begin(), edges[i].end());
  });
}

size_t configured_batch_size() {
  return config.max_batch_size > 0
             ? static_cast<size_t>(config.max_batch_size)
             : 0;
}

size_t configured_gpu_batch_threshold(size_t cpu_threads) {
  size_t threshold = max<size_t>(1, min<size_t>(8, cpu_threads / 2));
  const size_t batch_size = configured_batch_size();
  if (batch_size != 0)
    threshold = min(threshold, batch_size);
  return threshold;
}

} // namespace

vector<ProcessResult> process_batch(MeshList meshes, double concavity,
                                    int num_parts) {
  validate_parameters(meshes, concavity, num_parts);
  if (meshes.empty())
    return {};

  const size_t cpu_threads = configured_cpu_threads(meshes.size());
  BatchExecutor executor(cpu_threads);
  vector<BatchState> states(meshes.size());
  vector<ProcessResult> results(meshes.size());
  PipelineCoordinator coordinator(
      executor, states.size(), configured_gpu_batch_threshold(cpu_threads));
  OptixRuntime optix;
  CandidatePlaneRuntime candidate_planes;
  PlaneScoringRuntime plane_scoring;
  HausdorffRuntime hausdorff;
  DeviceMeshRuntime device_meshes;
  ClipBatchRuntime clip_batch;
  ComponentBatchRuntime component_batch;
  ConvexHullBatchRuntime convex_hulls;

  const uint32_t batch_seed = random_engine();
  for (size_t i = 0; i < states.size(); ++i) {
    seed_seq seed{batch_seed, static_cast<uint32_t>(i),
                  static_cast<uint32_t>(i >> 32)};
    states[i].random_engine.seed(seed);
  }

  function<void(shared_ptr<FinalizeWork>, double)> finish_state;
  function<void(shared_ptr<FinalizeWork>)> schedule_final_concavity;
  function<void(size_t)> schedule_finalize;
  function<void(size_t)> schedule_split;
  function<void(size_t)> schedule_advance;
  function<void(shared_ptr<ComponentWork>)> schedule_components;
  function<void(shared_ptr<SplitWork>)> schedule_clip;

  finish_state = [&](shared_ptr<FinalizeWork> work, double final_concavity) {
    coordinator.submit_cpu(
        [&, work = move(work), final_concavity]() mutable {
          BatchState &state = states[work->state_index];
          if (config.use_merging) {
            multimerge_ch(state.parts, work->hulls, final_concavity,
                          concavity);
          }

          state.part_cache.clear();
          work->hull_devices.clear();

          for (Mesh &hull : work->hulls)
            hull.unnormalize(state.original_bbox);
          for (Mesh &part : state.parts)
            part.unnormalize(state.original_bbox);

          MeshList output =
              config.return_parts ? move(state.parts) : move(work->hulls);
          const int output_count = static_cast<int>(output.size());
          ostringstream summary;
          summary << "Done. parts=" << output_count
                  << "  concavity=" << fixed << setprecision(4)
                  << final_concavity;
          log(work->state_index, summary.str());
          results[work->state_index] =
              {move(output), final_concavity, output_count};
          coordinator.complete_state();
        },
        true);
  };

  schedule_final_concavity = [&](shared_ptr<FinalizeWork> finalize_work) {
    coordinator.submit_cpu(
        [&, finalize_work = move(finalize_work)]() mutable {
          BatchState &state = states[finalize_work->state_index];
          if (state.parts.size() != finalize_work->hulls.size())
            throw logic_error("Final hulls are out of sync with batch state");
          if (state.parts.empty()) {
            finish_state(move(finalize_work), 0.0);
            return;
          }

          auto work = make_shared<HausdorffWork>();
          work->purpose = HausdorffPurpose::final_concavity;
          work->state_index = finalize_work->state_index;
          work->finalize_work = move(finalize_work);
          work->jobs.reserve(state.parts.size());
          for (size_t part_index = 0; part_index < state.parts.size();
               ++part_index) {
            work->jobs.push_back(prepare_hausdorff_job(
                state.parts[part_index], work->finalize_work->hulls[part_index],
                10000, true, state.random_engine));
          }
          coordinator.enqueue_hausdorff(move(work));
        },
        true);
  };

  schedule_finalize = [&](size_t state_index) {
    BatchState &state = states[state_index];
    if (state.finalizing)
      return;
    state.finalizing = true;

    auto work = make_shared<FinalizeWork>();
    work->state_index = state_index;
    work->hulls.resize(state.parts.size());
    work->hull_devices.resize(state.parts.size());
    log(state_index, "Computing convex hulls for " +
                         to_string(state.parts.size()) + " parts...");

    if (state.parts.size() != state.part_cache.size())
      throw logic_error("Part cache is out of sync during finalization");

    vector<size_t> missing_hulls;
    for (size_t part_index = 0; part_index < state.parts.size(); ++part_index) {
      if (state.part_cache[part_index].hull) {
        work->hulls[part_index] =
            move(*state.part_cache[part_index].hull);
        work->hull_devices[part_index] =
            move(state.part_cache[part_index].hull_device);
      } else {
        missing_hulls.push_back(part_index);
      }
    }
    if (missing_hulls.empty()) {
      schedule_final_concavity(move(work));
      return;
    }
    work->remaining.store(missing_hulls.size());
    for (size_t part_index : missing_hulls) {
      coordinator.submit_cpu(
          [&, work, part_index]() {
            states[work->state_index].parts[part_index].compute_ch(
                work->hulls[part_index], true);
            if (work->remaining.fetch_sub(1) == 1)
              schedule_final_concavity(work);
          },
          true);
    }
  };

  schedule_split = [&](size_t state_index) {
    coordinator.submit_cpu(
        [&, state_index]() {
          BatchState &state = states[state_index];
          int part_index = -1;
          if (config.score_mode == "edge") {
            part_index = get_part_with_highest_score(state.parts);
          } else if (config.score_mode == "concavity") {
            double max_concavity = -1.0;
            part_index =
                get_part_with_highest_cached_concavity(state, max_concavity);
            if (max_concavity < concavity) {
              log(state_index,
                  "Concavity " + to_string(max_concavity) +
                      " is below threshold; stopping.");
              state.finished = true;
              schedule_finalize(state_index);
              return;
            }
          }

          if (part_index < 0) {
            state.finished = true;
            schedule_finalize(state_index);
            return;
          }

          log(state_index,
              "Step " + to_string(state.iteration + 1) + "/" +
                  to_string(num_parts - 1) + ": splitting part " +
                  to_string(part_index) + " (" +
                  to_string(state.parts.size()) + " parts total).");

          Mesh part = move(state.parts[part_index]);
          PartCache part_cache = move(state.part_cache[part_index]);
          state.parts.erase(state.parts.begin() + part_index);
          state.part_cache.erase(state.part_cache.begin() + part_index);
          if (part.intersecting_edges.empty()) {
            state.parts.push_back(move(part));
            state.part_cache.push_back(move(part_cache));
            state.finished = true;
            log(state_index, "No more intersecting edges; stopping early.");
            schedule_finalize(state_index);
            return;
          }

          auto split = make_shared<SplitWork>(prepare_split_work(
              state_index, move(part), move(part_cache),
              state.random_engine));
          coordinator.enqueue_planes(move(split));
        },
        true);
  };

  schedule_advance = [&](size_t state_index) {
    coordinator.submit_cpu(
        [&, state_index]() {
          BatchState &state = states[state_index];
          if (state.finished || state.iteration >= num_parts - 1) {
            schedule_finalize(state_index);
            return;
          }

          if (config.score_mode == "edge") {
            schedule_split(state_index);
            return;
          }

          if (config.score_mode != "concavity") {
            schedule_split(state_index);
            return;
          }

          if (state.parts.size() != state.part_cache.size())
            throw logic_error("Part cache is out of sync with batch state");

          auto hull_work = make_shared<HullWork>();
          hull_work->state_index = state_index;
          for (size_t part_index = 0; part_index < state.parts.size();
               ++part_index) {
            PartCache &cache = state.part_cache[part_index];
            if (!cache.hull) {
              cache.hull.emplace();
              hull_work->part_indices.push_back(part_index);
            }
          }
          if (!hull_work->part_indices.empty()) {
            coordinator.enqueue_hulls(move(hull_work));
            return;
          }

          auto work = make_shared<HausdorffWork>();
          work->purpose = HausdorffPurpose::selection;
          work->state_index = state_index;
          for (size_t part_index = 0; part_index < state.parts.size();
               ++part_index) {
            PartCache &cache = state.part_cache[part_index];
            if (cache.selection_concavity)
              continue;
            work->part_indices.push_back(part_index);
            work->relative_volume_terms.push_back(
                compute_rv(state.parts[part_index], *cache.hull, 42) * 0.3);
            work->jobs.push_back(prepare_hausdorff_job(
                state.parts[part_index], *cache.hull, 3000, false,
                state.random_engine));
          }

          if (work->jobs.empty()) {
            schedule_split(state_index);
            return;
          }
          coordinator.enqueue_hausdorff(move(work));
        },
        true);
  };

  schedule_components = [&](shared_ptr<ComponentWork> work) {
    coordinator.submit_cpu(
        [&, work = move(work)]() mutable {
          BatchState &state = states[work->state_index];
          separate_disjoint_prepared(work->parts, work->labels);

          if (work->initial) {
            state.parts = move(work->parts);
            state.part_cache.resize(state.parts.size());
            auto intersections = make_shared<IntersectionWork>();
            intersections->state_index = work->state_index;
            intersections->initial = true;
            intersections->requests.reserve(state.parts.size());
            for (Mesh &part : state.parts)
              intersections->requests.emplace_back(&part, &state.cage);
            coordinator.enqueue_intersections(move(intersections));
            return;
          }

          auto intersections = make_shared<IntersectionWork>();
          intersections->state_index = work->state_index;
          for (Mesh &part : work->parts) {
            if (part.vertices.size() < 10)
              continue;
            PendingPart pending;
            pending.state_index = work->state_index;
            pending.part = move(part);
            pending.cage = pending.part.copy();
            intersections->pending_parts.push_back(move(pending));
          }

          if (intersections->pending_parts.empty()) {
            ++state.iteration;
            schedule_advance(work->state_index);
            return;
          }

          intersections->requests.reserve(
              intersections->pending_parts.size());
          for (PendingPart &pending : intersections->pending_parts) {
            intersections->requests.emplace_back(&pending.part,
                                                  &pending.cage);
          }

          auto remaining = make_shared<atomic<size_t>>(
              intersections->pending_parts.size());
          for (size_t part_index = 0;
               part_index < intersections->pending_parts.size();
               ++part_index) {
            coordinator.submit_cpu(
                [&, intersections, remaining, part_index]() {
                  manifold_preprocess(
                      intersections->pending_parts[part_index].cage, 40,
                      0.02);
                  if (remaining->fetch_sub(1) == 1)
                    coordinator.enqueue_intersections(intersections);
                },
                true);
          }
        },
        true);
  };

  schedule_clip = [&](shared_ptr<SplitWork> split) {
    coordinator.submit_cpu(
        [&, split = move(split)]() mutable {
          BatchState &state = states[split->state_index];
          int *first_map = nullptr;
          int *second_map = nullptr;
          MeshList new_parts;
          if (split->prepared_clip.empty()) {
            new_parts = clip(split->part,
                             split->planes[split->selected_plane], first_map,
                             second_map);
          } else {
            new_parts = clip_prepared(
                split->part, split->planes[split->selected_plane], first_map,
                second_map, split->prepared_clip);
          }
          if (new_parts.size() < 2) {
            delete[] first_map;
            delete[] second_map;
            state.parts.push_back(move(split->part));
            state.part_cache.push_back(move(split->part_cache));
            ++state.iteration;
            schedule_advance(split->state_index);
            return;
          }

          propagate_existing_edges(split->part, first_map, second_map,
                                   new_parts);
          delete[] first_map;
          delete[] second_map;

          auto components = make_shared<ComponentWork>();
          components->state_index = split->state_index;
          components->parts = move(new_parts);
          components->labels.resize(components->parts.size());
          coordinator.enqueue_components(move(components));
        },
        true);
  };

  for (size_t state_index = 0; state_index < states.size(); ++state_index) {
    coordinator.submit_cpu([&, state_index]() {
      initialize_state(move(meshes[state_index]), state_index,
                       states[state_index]);
      auto initial = make_shared<ComponentWork>();
      initial->state_index = state_index;
      initial->initial = true;
      initial->parts = move(states[state_index].parts);
      initial->labels.resize(initial->parts.size());
      coordinator.enqueue_components(move(initial));
    });
  }

  try {
    while (true) {
      GpuWork gpu_work = coordinator.wait_for_gpu_work();
      if (gpu_work.kind == GpuWorkKind::complete)
        break;

      if (gpu_work.kind == GpuWorkKind::intersections) {
        vector<pair<Mesh *, Mesh *>> requests;
        size_t request_count = 0;
        for (const shared_ptr<IntersectionWork> &work :
             gpu_work.intersections) {
          request_count += work->requests.size();
        }
        requests.reserve(request_count);
        for (const shared_ptr<IntersectionWork> &work :
             gpu_work.intersections) {
          requests.insert(requests.end(), work->requests.begin(),
                          work->requests.end());
        }

        auto edges = compute_intersection_matrices(
            requests, optix, configured_batch_size(),
            config.batch_memory_fraction, &executor);
        append_intersections(requests, edges, executor);

        for (shared_ptr<IntersectionWork> &work :
             gpu_work.intersections) {
          BatchState &state = states[work->state_index];
          if (work->initial) {
            log(work->state_index,
                "Starting decomposition (max parts=" +
                    to_string(num_parts) + ", concavity threshold=" +
                    to_string(concavity) + ", mode=" + config.score_mode +
                    ").");
          } else {
            for (PendingPart &pending : work->pending_parts) {
              state.parts.push_back(move(pending.part));
              state.part_cache.emplace_back();
            }
            ++state.iteration;
          }
          schedule_advance(work->state_index);
        }
        continue;
      }

      if (gpu_work.kind == GpuWorkKind::hulls) {
        size_t input_count = 0;
        for (const shared_ptr<HullWork> &work : gpu_work.hulls)
          input_count += work->part_indices.size();
        vector<ConvexHullBatchInput> inputs;
        inputs.reserve(input_count);
        for (shared_ptr<HullWork> &work : gpu_work.hulls) {
          BatchState &state = states[work->state_index];
          for (size_t part_index : work->part_indices) {
            if (part_index >= state.parts.size() ||
                part_index >= state.part_cache.size()) {
              throw logic_error("Hull part index is out of range");
            }
            PartCache &cache = state.part_cache[part_index];
            if (!cache.device) {
              cache.device = device_meshes.try_upload(
                  state.parts[part_index], config.batch_memory_fraction);
            }
            if (!cache.hull)
              throw logic_error("Selection hull destination is missing");
            inputs.push_back({&state.parts[part_index],
                              cache.device.get(), &*cache.hull, true});
          }
        }
        coordinator.submit_cpu(
            [&, inputs = move(inputs),
             works = move(gpu_work.hulls)]() mutable {
              try {
                compute_convex_hulls_batch(
                    inputs, convex_hulls, configured_batch_size(),
                    config.batch_memory_fraction);
                for (shared_ptr<HullWork> &work : works)
                  schedule_advance(work->state_index);
                coordinator.complete_hull_batch();
              } catch (...) {
                coordinator.complete_hull_batch();
                throw;
              }
            },
            true);
        continue;
      }

      if (gpu_work.kind == GpuWorkKind::hausdorff) {
        for (shared_ptr<HausdorffWork> &work : gpu_work.hausdorff) {
          BatchState &state = states[work->state_index];
          if (work->purpose == HausdorffPurpose::selection) {
            if (work->part_indices.size() != work->jobs.size())
              throw logic_error("Selection Hausdorff work is out of sync");
            for (size_t job_index = 0; job_index < work->jobs.size();
                 ++job_index) {
              const size_t part_index = work->part_indices[job_index];
              if (part_index >= state.parts.size() ||
                  part_index >= state.part_cache.size()) {
                throw logic_error("Selection part index is out of range");
              }
              PartCache &cache = state.part_cache[part_index];
              if (!cache.device) {
                cache.device = device_meshes.try_upload(
                    state.parts[part_index], config.batch_memory_fraction);
              }
              if (!cache.hull_device) {
                cache.hull_device = device_meshes.try_upload(
                    *cache.hull, config.batch_memory_fraction);
              }
              attach_hausdorff_device_meshes(
                  work->jobs[job_index], cache.device, cache.hull_device);
            }
            continue;
          }

          if (!work->finalize_work ||
              work->jobs.size() != state.parts.size() ||
              work->finalize_work->hulls.size() != state.parts.size() ||
              work->finalize_work->hull_devices.size() !=
                  state.parts.size()) {
            throw logic_error("Final Hausdorff work is out of sync");
          }
          for (size_t part_index = 0; part_index < state.parts.size();
               ++part_index) {
            PartCache &cache = state.part_cache[part_index];
            if (!cache.device) {
              cache.device = device_meshes.try_upload(
                  state.parts[part_index], config.batch_memory_fraction);
            }
            shared_ptr<DeviceMesh> &hull_device =
                work->finalize_work->hull_devices[part_index];
            if (!hull_device) {
              hull_device = device_meshes.try_upload(
                  work->finalize_work->hulls[part_index],
                  config.batch_memory_fraction);
            }
            attach_hausdorff_device_meshes(work->jobs[part_index],
                                            cache.device, hull_device);
          }
        }

        size_t job_count = 0;
        for (const shared_ptr<HausdorffWork> &work : gpu_work.hausdorff)
          job_count += work->jobs.size();
        vector<PreparedHausdorffJob *> jobs;
        jobs.reserve(job_count);
        for (shared_ptr<HausdorffWork> &work : gpu_work.hausdorff) {
          for (PreparedHausdorffJob &job : work->jobs)
            jobs.push_back(&job);
        }
        evaluate_hausdorff_batch(jobs, hausdorff, configured_batch_size(),
                                 config.batch_memory_fraction);

        for (shared_ptr<HausdorffWork> &work : gpu_work.hausdorff) {
          if (work->purpose == HausdorffPurpose::selection) {
            if (work->part_indices.size() != work->jobs.size() ||
                work->relative_volume_terms.size() != work->jobs.size()) {
              throw logic_error("Selection Hausdorff work is out of sync");
            }
            BatchState &state = states[work->state_index];
            for (size_t job_index = 0; job_index < work->jobs.size();
                 ++job_index) {
              const size_t part_index = work->part_indices[job_index];
              if (part_index >= state.part_cache.size())
                throw logic_error("Selection part index is out of range");
              state.part_cache[part_index].selection_concavity =
                  max(work->relative_volume_terms[job_index],
                      work->jobs[job_index].result);
            }
            schedule_split(work->state_index);
            continue;
          }

          if (!work->finalize_work)
            throw logic_error("Final Hausdorff work has no hulls");
          double final_concavity = 0.0;
          for (const PreparedHausdorffJob &job : work->jobs)
            final_concavity = max(final_concavity, job.result);
          finish_state(move(work->finalize_work), final_concavity);
        }
        continue;
      }

      if (gpu_work.kind == GpuWorkKind::components) {
        size_t input_count = 0;
        for (const shared_ptr<ComponentWork> &work : gpu_work.components)
          input_count += work->parts.size();
        vector<ComponentBatchInput> inputs;
        inputs.reserve(input_count);
        for (shared_ptr<ComponentWork> &work : gpu_work.components) {
          if (work->labels.size() != work->parts.size())
            throw logic_error("Component work is out of sync");
          for (size_t part_index = 0; part_index < work->parts.size();
               ++part_index) {
            inputs.push_back(
                {&work->parts[part_index], &work->labels[part_index]});
          }
        }
        label_components_batch(inputs, component_batch,
                               configured_batch_size(),
                               config.batch_memory_fraction);
        for (shared_ptr<ComponentWork> &work : gpu_work.components)
          schedule_components(move(work));
        continue;
      }

      vector<CandidatePlaneInput> candidate_inputs;
      candidate_inputs.reserve(gpu_work.planes.size());
      for (shared_ptr<SplitWork> &split : gpu_work.planes) {
        if (split->part.vertices.size() >
                static_cast<size_t>(numeric_limits<int>::max()) ||
            split->part.intersecting_edges.size() >
                static_cast<size_t>(numeric_limits<int>::max())) {
          throw overflow_error("Candidate-plane input is too large");
        }
        if (!split->part_cache.device) {
          split->part_cache.device = device_meshes.try_upload(
              split->part, config.batch_memory_fraction);
        }
        candidate_inputs.push_back(
            {&split->part, split->part_cache.device.get(),
             split->sampled_edges.data(), split->sampled_edges.size(),
             kCandidatePlaneCount, &split->planes,
             &split->candidate_attempts});
      }
      generate_candidate_planes_batch(
          candidate_inputs, candidate_planes, configured_batch_size(),
          config.batch_memory_fraction);

      vector<shared_ptr<SplitWork>> active_splits;
      active_splits.reserve(gpu_work.planes.size());
      for (shared_ptr<SplitWork> &split : gpu_work.planes) {
        BatchState &state = states[split->state_index];
        finish_split_work(*split, state.flat_surface_planes,
                          state.random_engine);
        if (split->planes.empty()) {
          state.parts.push_back(move(split->part));
          state.part_cache.push_back(move(split->part_cache));
          state.finished = true;
          schedule_finalize(split->state_index);
          continue;
        }
        active_splits.push_back(move(split));
      }
      if (active_splits.empty())
        continue;

      vector<PlaneScoreInput> score_inputs;
      score_inputs.reserve(active_splits.size());
      for (shared_ptr<SplitWork> &split : active_splits) {
        if (split->planes.size() >
            static_cast<size_t>(numeric_limits<int>::max())) {
          throw overflow_error("Plane scoring input is too large");
        }
        score_inputs.push_back(
            {split->host_planes.data(), split->host_points.data(),
             split->host_edges.data(), split->scores.data(),
             static_cast<int>(split->planes.size()),
             static_cast<int>(split->part.vertices.size()),
             static_cast<int>(split->part.intersecting_edges.size()),
             split->part_cache.device.get()});
      }
      classify_and_rate_planes_batch(score_inputs, plane_scoring,
                                     configured_batch_size(),
                                     config.batch_memory_fraction);
      vector<ClipBatchInput> clip_inputs;
      clip_inputs.reserve(active_splits.size());
      for (shared_ptr<SplitWork> &split : active_splits) {
        for (size_t index = split->flat_surface_offset;
             index < split->scores.size(); ++index) {
          split->scores[index] *=
              static_cast<float>(config.flat_surface_k);
        }
        split->selected_plane =
            max_element(split->scores.begin(), split->scores.end()) -
            split->scores.begin();
        if (split->part_cache.device) {
          clip_inputs.push_back(
              {split->part_cache.device.get(),
               split->planes[split->selected_plane],
               &split->prepared_clip});
        }
      }
      prepare_clip_batch(clip_inputs, clip_batch, configured_batch_size(),
                         config.batch_memory_fraction);
      for (shared_ptr<SplitWork> &split : active_splits)
        schedule_clip(move(split));
    }
  } catch (...) {
    coordinator.set_error(current_exception());
    coordinator.wait_for_gpu_work();
    throw;
  }

  return results;
}

} // namespace neural_acd
