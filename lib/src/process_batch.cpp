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
#include <cstdlib>
#include <deque>
#include <device_mesh.hpp>
#include <exception>
#include <flat_surfaces_batch.hpp>
#include <functional>
#include <hausdorff_batch.hpp>
#include <iomanip>
#include <intersections.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <merge_batch.hpp>
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

enum class ProfileStage {
  preprocess,
  preprocess_copy,
  preprocess_marshal,
  preprocess_sdf,
  preprocess_surface,
  preprocess_output,
  flat_surfaces,
  components_gpu,
  components_cpu,
  intersections,
  selection_hulls,
  hausdorff,
  candidates,
  plane_scoring,
  plane_selection,
  clip_prepare,
  split_fused,
  clip_construct,
  child_cage,
  final_hulls_cpu,
  merges,
  count
};

constexpr size_t kProfileStageCount =
    static_cast<size_t>(ProfileStage::count);

class StageProfiler {
public:
  StageProfiler() {
    const char *value = getenv("VISACD_STAGE_TIMING");
    enabled_ = value && *value && string(value) != "0";
    for (auto &value : nanoseconds_)
      value.store(0, memory_order_relaxed);
    for (auto &value : calls_)
      value.store(0, memory_order_relaxed);
    for (auto &value : items_)
      value.store(0, memory_order_relaxed);
  }

  bool enabled() const { return enabled_; }

  void record(ProfileStage stage, chrono::steady_clock::duration duration,
              size_t items) {
    const size_t index = static_cast<size_t>(stage);
    nanoseconds_[index].fetch_add(
        chrono::duration_cast<chrono::nanoseconds>(duration).count(),
        memory_order_relaxed);
    calls_[index].fetch_add(1, memory_order_relaxed);
    items_[index].fetch_add(items, memory_order_relaxed);
  }

  void report(chrono::steady_clock::duration total, size_t mesh_count) const {
    if (!enabled_)
      return;
    static const array<const char *, kProfileStageCount> names = {
        "preprocess",         "preprocess_copy",
        "preprocess_marshal", "preprocess_sdf",
        "preprocess_surface", "preprocess_output",
        "flat_surfaces",      "components_gpu",
        "components_cpu",     "intersections",
        "selection_hulls",    "hausdorff",
        "candidates",         "plane_scoring",
        "plane_selection",    "clip_prepare",
        "split_fused",        "clip_construct",
        "child_cage",         "final_hulls_cpu",
        "merges"};

    ostringstream output;
    output << fixed << setprecision(3)
           << "[visacd stages] total_ms="
           << chrono::duration<double, milli>(total).count()
           << " meshes=" << mesh_count << '\n';
    for (size_t index = 0; index < names.size(); ++index) {
      const unsigned long long calls =
          calls_[index].load(memory_order_relaxed);
      if (calls == 0)
        continue;
      const double milliseconds =
          static_cast<double>(
              nanoseconds_[index].load(memory_order_relaxed)) /
          1.0e6;
      output << "[visacd stages] stage=" << names[index]
             << " calls=" << calls
             << " items=" << items_[index].load(memory_order_relaxed)
             << " wall_ms=" << milliseconds << '\n';
    }
    cerr << output.str();
    cerr.flush();
  }

private:
  bool enabled_ = false;
  array<atomic<long long>, kProfileStageCount> nanoseconds_;
  array<atomic<unsigned long long>, kProfileStageCount> calls_;
  array<atomic<unsigned long long>, kProfileStageCount> items_;
};

class StageTimer {
public:
  StageTimer(StageProfiler &profiler, ProfileStage stage, size_t items)
      : profiler_(profiler.enabled() ? &profiler : nullptr), stage_(stage),
        items_(items) {
    if (profiler_)
      start_ = chrono::steady_clock::now();
  }

  ~StageTimer() {
    if (profiler_)
      profiler_->record(stage_, chrono::steady_clock::now() - start_, items_);
  }

  StageTimer(const StageTimer &) = delete;
  StageTimer &operator=(const StageTimer &) = delete;

private:
  StageProfiler *profiler_;
  ProfileStage stage_;
  size_t items_;
  chrono::steady_clock::time_point start_;
};

void profiled_manifold_preprocess(Mesh &mesh, double scale,
                                  double level_set,
                                  StageProfiler &profiler) {
  if (!profiler.enabled()) {
    manifold_preprocess(mesh, scale, level_set);
    return;
  }

  ManifoldPreprocessMetrics metrics;
  manifold_preprocess(mesh, scale, level_set, &metrics);
  profiler.record(ProfileStage::preprocess_copy,
                  chrono::nanoseconds(metrics.copy_input_ns),
                  metrics.input_triangles);
  profiler.record(ProfileStage::preprocess_marshal,
                  chrono::nanoseconds(metrics.marshal_input_ns),
                  metrics.input_triangles);
  profiler.record(ProfileStage::preprocess_sdf,
                  chrono::nanoseconds(metrics.mesh_to_sdf_ns),
                  metrics.active_voxels);
  profiler.record(ProfileStage::preprocess_surface,
                  chrono::nanoseconds(metrics.volume_to_mesh_ns),
                  metrics.active_voxels);
  profiler.record(ProfileStage::preprocess_output,
                  chrono::nanoseconds(metrics.marshal_output_ns),
                  metrics.output_triangles);
}

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
  vector<unsigned int> sampled_edges;
  size_t candidate_attempts = 0;
  Plane selected_plane;
  bool has_selected_plane = false;
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
  vector<MeshList> separated;
  shared_ptr<DeviceMesh> projected_edge_source;
  vector<vector<int>> projected_vertex_maps;
};

struct FinalizeWork {
  size_t state_index;
  MeshList hulls;
  vector<shared_ptr<DeviceMesh>> hull_devices;
  vector<double> part_hausdorff;
  atomic<size_t> remaining{0};
};

struct HullWork {
  size_t state_index;
  vector<size_t> part_indices;
  size_t estimated_size = 0;
};

struct FlatSurfaceWork {
  size_t state_index;
  vector<Surface> surfaces;
  size_t estimated_size = 0;
};

struct MergeWork {
  size_t state_index;
  double final_concavity;
  size_t initial_hull_count;
  shared_ptr<FinalizeWork> finalize_work;
  vector<shared_ptr<DeviceMesh>> part_devices;
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
  flat_surfaces,
  merges,
  complete
};

constexpr size_t kGpuWorkKindCount =
    static_cast<size_t>(GpuWorkKind::complete);

struct GpuWork {
  GpuWorkKind kind;
  size_t lane = 0;
  bool shared_memory_budget = false;
  vector<shared_ptr<IntersectionWork>> intersections;
  vector<shared_ptr<SplitWork>> planes;
  vector<shared_ptr<HausdorffWork>> hausdorff;
  vector<shared_ptr<ComponentWork>> components;
  vector<shared_ptr<HullWork>> hulls;
  vector<shared_ptr<FlatSurfaceWork>> flat_surfaces;
  vector<shared_ptr<MergeWork>> merges;
};

size_t saturating_add(size_t first, size_t second) {
  return second > numeric_limits<size_t>::max() - first
             ? numeric_limits<size_t>::max()
             : first + second;
}

size_t saturating_multiply(size_t value, size_t multiplier) {
  return value != 0 &&
                 multiplier > numeric_limits<size_t>::max() / value
             ? numeric_limits<size_t>::max()
             : value * multiplier;
}

size_t mesh_work_size(const Mesh &mesh) {
  size_t size = mesh.vertices.size();
  size = saturating_add(
      size, saturating_multiply(mesh.triangles.size(), size_t{3}));
  return saturating_add(
      size, saturating_multiply(mesh.intersecting_edges.size(), size_t{2}));
}

size_t work_size_bucket(size_t size) {
  size_t bucket = 0;
  while (size > 1) {
    size >>= 1;
    ++bucket;
  }
  return bucket;
}

bool work_bucketing_enabled() {
  const char *disabled = getenv("VISACD_DISABLE_WORK_BUCKETING");
  return !disabled || !*disabled || string(disabled) == "0";
}

bool double_buffering_enabled() {
  const char *disabled = getenv("VISACD_DISABLE_DOUBLE_BUFFERING");
  return !disabled || !*disabled || string(disabled) == "0";
}

template <typename Work, typename SizeFunction>
void bucket_work(vector<shared_ptr<Work>> &work,
                 SizeFunction size_function) {
  if (work.size() < 2 || !work_bucketing_enabled())
    return;
  stable_sort(work.begin(), work.end(), [&](const auto &first,
                                            const auto &second) {
    return work_size_bucket(size_function(*first)) >
           work_size_bucket(size_function(*second));
  });
}

template <typename Work, typename SizeFunction>
bool queue_spans_work_size_buckets(
    const deque<shared_ptr<Work>> &work, SizeFunction size_function) {
  if (work.size() < 2)
    return false;
  const size_t first_bucket = work_size_bucket(size_function(*work.front()));
  for (size_t index = 1; index < work.size(); ++index) {
    if (work_size_bucket(size_function(*work[index])) != first_bucket)
      return true;
  }
  return false;
}

template <typename Work>
void split_work_wave(vector<shared_ptr<Work>> &work,
                     deque<shared_ptr<Work>> &queue, bool split) {
  if (!split || work.size() < 2)
    return;
  const size_t wave_size = (work.size() + 1) / 2;
  for (size_t index = wave_size; index < work.size(); ++index)
    queue.push_back(move(work[index]));
  work.resize(wave_size);
}

size_t intersection_work_size(const IntersectionWork &work) {
  size_t size = 0;
  for (const auto &request : work.requests) {
    if (request.first)
      size = saturating_add(size, mesh_work_size(*request.first));
    if (request.second)
      size = saturating_add(size, mesh_work_size(*request.second));
  }
  return size;
}

size_t split_work_size(const SplitWork &work) {
  return saturating_add(mesh_work_size(work.part),
                        work.sampled_edges.size());
}

size_t hausdorff_work_size(const HausdorffWork &work) {
  size_t size = 0;
  for (const PreparedHausdorffJob &job : work.jobs) {
    for (const PreparedHausdorffDirection &direction : job.directions) {
      size = saturating_add(size, direction.queries.size());
      size = saturating_add(size, direction.candidate_triangles.size());
      if (direction.target && !direction.target_device)
        size = saturating_add(size, mesh_work_size(*direction.target));
    }
  }
  return size;
}

size_t component_work_size(const ComponentWork &work) {
  size_t size = 0;
  for (const Mesh &part : work.parts)
    size = saturating_add(size, mesh_work_size(part));
  return size;
}

size_t merge_work_size(const MergeWork &work) {
  size_t size = 0;
  if (!work.finalize_work)
    return size;
  for (const Mesh &hull : work.finalize_work->hulls)
    size = saturating_add(size, mesh_work_size(hull));
  return size;
}

class PipelineCoordinator {
public:
  PipelineCoordinator(BatchExecutor &executor,
                      BatchExecutor &gpu_executor,
                      size_t total_states, size_t gpu_batch_threshold)
      : executor_(executor), gpu_executor_(gpu_executor),
        total_states_(total_states),
        gpu_batch_threshold_(max<size_t>(1, gpu_batch_threshold)) {}

  void submit_cpu(function<void()> task, bool priority = false) {
    submit_task(executor_, move(task), priority);
  }

  void submit_gpu(function<void()> task) {
    submit_task(gpu_executor_, move(task), false);
  }

  void complete_gpu_batch(GpuWorkKind kind, size_t lane) {
    lock_guard<mutex> lock(mutex_);
    const size_t index = static_cast<size_t>(kind);
    if (index >= gpu_busy_.size() || lane >= gpu_busy_[index].size())
      return;
    gpu_busy_[index][lane] = false;
    condition_.notify_one();
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

  void enqueue_flat_surfaces(shared_ptr<FlatSurfaceWork> work) {
    lock_guard<mutex> lock(mutex_);
    if (error_)
      return;
    ++flat_surface_request_count_;
    flat_surface_queue_.push_back(move(work));
    condition_.notify_one();
  }

  void enqueue_merge(shared_ptr<MergeWork> work) {
    lock_guard<mutex> lock(mutex_);
    if (error_)
      return;
    ++merge_request_count_;
    merge_queue_.push_back(move(work));
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
        return error_ || completed() ||
               (gpu_lane_available(0) && !intersection_queue_.empty()) ||
               (gpu_lane_available(1) && !plane_queue_.empty()) ||
               (gpu_lane_available(2) && !hausdorff_queue_.empty()) ||
               (gpu_lane_available(3) && !component_queue_.empty()) ||
               (gpu_lane_available(4) && !hull_queue_.empty()) ||
               (gpu_lane_available(5) && !flat_surface_queue_.empty()) ||
               (gpu_lane_available(6) && !merge_queue_.empty());
      });

      if (error_) {
        if (active_tasks_ != 0) {
          condition_.wait(lock, [this]() { return active_tasks_ == 0; });
        }
        rethrow_exception(error_);
      }
      if (completed())
        return GpuWork{GpuWorkKind::complete};

      if (!gpu_batch_ready() && active_tasks_ != 0) {
        condition_.wait_for(lock, chrono::milliseconds(1), [this]() {
          return error_ || completed() || gpu_batch_ready() ||
                 active_tasks_ == 0;
        });
        if (error_)
          continue;
        if (completed())
          return GpuWork{GpuWorkKind::complete};
      }

      const array<bool, kGpuWorkKindCount> available = {
          gpu_lane_available(0) && !intersection_queue_.empty(),
          gpu_lane_available(1) && !plane_queue_.empty(),
          gpu_lane_available(2) && !hausdorff_queue_.empty(),
          gpu_lane_available(3) && !component_queue_.empty(),
          gpu_lane_available(4) && !hull_queue_.empty(),
          gpu_lane_available(5) && !flat_surface_queue_.empty(),
          gpu_lane_available(6) && !merge_queue_.empty()};
      size_t selected = 0;
      for (; selected < available.size(); ++selected) {
        const size_t candidate =
            (next_gpu_kind_ + selected) % available.size();
        if (available[candidate]) {
          selected = candidate;
          break;
        }
      }
      next_gpu_kind_ = (selected + 1) % available.size();
      const size_t free_lanes = free_gpu_lanes(selected);
      const bool joining_split_wave = gpu_split_pending_[selected];
      const size_t lane = first_free_gpu_lane(selected);
      gpu_busy_[selected][lane] = true;
      const bool split_wave =
          !joining_split_wave && free_lanes > 1 &&
          double_buffer_worthwhile(selected);
      gpu_split_pending_[selected] = split_wave;
      const bool shared_memory_budget =
          gpu_lane_limit(selected) > 1 &&
          (split_wave || joining_split_wave);

      if (selected == 0) {
        GpuWork result{GpuWorkKind::intersections};
        result.lane = lane;
        result.shared_memory_budget = shared_memory_budget;
        result.intersections.reserve(intersection_queue_.size());
        while (!intersection_queue_.empty()) {
          shared_ptr<IntersectionWork> work =
              move(intersection_queue_.front());
          intersection_queue_.pop_front();
          result.intersections.push_back(move(work));
        }
        bucket_work(result.intersections, intersection_work_size);
        split_work_wave(result.intersections, intersection_queue_,
                        split_wave);
        for (const auto &work : result.intersections)
          intersection_request_count_ -= work->requests.size();
        return result;
      }

      if (selected == 1) {
        GpuWork result{GpuWorkKind::planes};
        result.lane = lane;
        result.shared_memory_budget = shared_memory_budget;
        result.planes.reserve(plane_queue_.size());
        while (!plane_queue_.empty()) {
          result.planes.push_back(move(plane_queue_.front()));
          plane_queue_.pop_front();
        }
        bucket_work(result.planes, split_work_size);
        split_work_wave(result.planes, plane_queue_, split_wave);
        return result;
      }

      if (selected == 2) {
        GpuWork result{GpuWorkKind::hausdorff};
        result.lane = lane;
        result.shared_memory_budget = shared_memory_budget;
        result.hausdorff.reserve(hausdorff_queue_.size());
        while (!hausdorff_queue_.empty()) {
          shared_ptr<HausdorffWork> work = move(hausdorff_queue_.front());
          hausdorff_queue_.pop_front();
          result.hausdorff.push_back(move(work));
        }
        bucket_work(result.hausdorff, hausdorff_work_size);
        split_work_wave(result.hausdorff, hausdorff_queue_, split_wave);
        for (const auto &work : result.hausdorff)
          hausdorff_request_count_ -= work->jobs.size();
        return result;
      }

      if (selected == 3) {
        GpuWork result{GpuWorkKind::components};
        result.lane = lane;
        result.shared_memory_budget = shared_memory_budget;
        result.components.reserve(component_queue_.size());
        while (!component_queue_.empty()) {
          shared_ptr<ComponentWork> work = move(component_queue_.front());
          component_queue_.pop_front();
          result.components.push_back(move(work));
        }
        bucket_work(result.components, component_work_size);
        split_work_wave(result.components, component_queue_, split_wave);
        for (const auto &work : result.components)
          component_request_count_ -= work->parts.size();
        return result;
      }

      if (selected == 4) {
        GpuWork result{GpuWorkKind::hulls};
        result.lane = lane;
        result.shared_memory_budget = shared_memory_budget;
        result.hulls.reserve(hull_queue_.size());
        while (!hull_queue_.empty()) {
          shared_ptr<HullWork> work = move(hull_queue_.front());
          hull_queue_.pop_front();
          result.hulls.push_back(move(work));
        }
        bucket_work(result.hulls, [](const HullWork &work) {
          return work.estimated_size;
        });
        split_work_wave(result.hulls, hull_queue_, split_wave);
        for (const auto &work : result.hulls)
          hull_request_count_ -= work->part_indices.size();
        return result;
      }

      if (selected == 5) {
        GpuWork result{GpuWorkKind::flat_surfaces};
        result.lane = lane;
        result.shared_memory_budget = shared_memory_budget;
        result.flat_surfaces.reserve(flat_surface_queue_.size());
        while (!flat_surface_queue_.empty()) {
          result.flat_surfaces.push_back(move(flat_surface_queue_.front()));
          flat_surface_queue_.pop_front();
        }
        bucket_work(result.flat_surfaces, [](const FlatSurfaceWork &work) {
          return work.estimated_size;
        });
        split_work_wave(result.flat_surfaces, flat_surface_queue_,
                        split_wave);
        flat_surface_request_count_ -= result.flat_surfaces.size();
        return result;
      }

      GpuWork result{GpuWorkKind::merges};
      result.lane = lane;
      result.shared_memory_budget = shared_memory_budget;
      result.merges.reserve(merge_queue_.size());
      while (!merge_queue_.empty()) {
        result.merges.push_back(move(merge_queue_.front()));
        merge_queue_.pop_front();
      }
      bucket_work(result.merges, merge_work_size);
      split_work_wave(result.merges, merge_queue_, split_wave);
      merge_request_count_ -= result.merges.size();
      return result;
    }
  }

private:
  void submit_task(BatchExecutor &executor, function<void()> task,
                   bool priority) {
    {
      lock_guard<mutex> lock(mutex_);
      if (error_)
        return;
      ++active_tasks_;
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
      finish_task(error);
    };

    try {
      if (priority)
        executor.submit_priority(move(wrapped));
      else
        executor.submit(move(wrapped));
    } catch (...) {
      finish_task(current_exception());
    }
  }
  bool completed() const {
    return completed_states_ == total_states_ && active_tasks_ == 0;
  }

  size_t gpu_lane_limit(size_t kind) const {
    return kind == static_cast<size_t>(GpuWorkKind::intersections) ||
                   !double_buffering_enabled()
               ? 1
               : 2;
  }

  bool double_buffer_worthwhile(size_t kind) const {
    if (!double_buffering_enabled())
      return false;
    switch (static_cast<GpuWorkKind>(kind)) {
    case GpuWorkKind::intersections:
      return queue_spans_work_size_buckets(
          intersection_queue_, intersection_work_size);
    case GpuWorkKind::planes:
      return queue_spans_work_size_buckets(plane_queue_, split_work_size);
    case GpuWorkKind::hausdorff:
      return queue_spans_work_size_buckets(
          hausdorff_queue_, hausdorff_work_size);
    case GpuWorkKind::components:
      return queue_spans_work_size_buckets(
          component_queue_, component_work_size);
    case GpuWorkKind::hulls:
      return queue_spans_work_size_buckets(
          hull_queue_, [](const HullWork &work) {
            return work.estimated_size;
          });
    case GpuWorkKind::flat_surfaces:
      return queue_spans_work_size_buckets(
          flat_surface_queue_, [](const FlatSurfaceWork &work) {
            return work.estimated_size;
          });
    case GpuWorkKind::merges:
      return queue_spans_work_size_buckets(merge_queue_, merge_work_size);
    case GpuWorkKind::complete:
      return false;
    }
    return false;
  }

  size_t free_gpu_lanes(size_t kind) const {
    size_t free = 0;
    for (size_t lane = 0; lane < gpu_lane_limit(kind); ++lane) {
      if (!gpu_busy_[kind][lane])
        ++free;
    }
    return free;
  }

  bool gpu_lane_available(size_t kind) const {
    const size_t free_lanes = free_gpu_lanes(kind);
    return free_lanes == gpu_lane_limit(kind) ||
           (free_lanes != 0 && gpu_split_pending_[kind]);
  }

  size_t first_free_gpu_lane(size_t kind) const {
    for (size_t lane = 0; lane < gpu_lane_limit(kind); ++lane) {
      if (!gpu_busy_[kind][lane])
        return lane;
    }
    throw logic_error("No free GPU work lane");
  }

  bool gpu_batch_ready() const {
    return (gpu_lane_available(0) &&
            intersection_request_count_ >= gpu_batch_threshold_) ||
           (gpu_lane_available(1) &&
            plane_queue_.size() >= gpu_batch_threshold_) ||
           (gpu_lane_available(2) &&
            hausdorff_request_count_ >= gpu_batch_threshold_) ||
           (gpu_lane_available(3) &&
            component_request_count_ >= gpu_batch_threshold_) ||
           (gpu_lane_available(4) &&
            hull_request_count_ >= gpu_batch_threshold_) ||
           (gpu_lane_available(5) &&
            flat_surface_request_count_ >= gpu_batch_threshold_) ||
           (gpu_lane_available(6) &&
            merge_request_count_ >= gpu_batch_threshold_);
  }

  void record_error(exception_ptr error) {
    if (!error_)
      error_ = error;
    intersection_queue_.clear();
    plane_queue_.clear();
    hausdorff_queue_.clear();
    component_queue_.clear();
    hull_queue_.clear();
    flat_surface_queue_.clear();
    merge_queue_.clear();
    intersection_request_count_ = 0;
    hausdorff_request_count_ = 0;
    component_request_count_ = 0;
    hull_request_count_ = 0;
    flat_surface_request_count_ = 0;
    merge_request_count_ = 0;
    for (auto &lanes : gpu_busy_)
      lanes.fill(false);
    gpu_split_pending_.fill(false);
    condition_.notify_all();
  }

  void finish_task(exception_ptr error) {
    lock_guard<mutex> lock(mutex_);
    if (error)
      record_error(error);
    --active_tasks_;
    condition_.notify_all();
  }

  BatchExecutor &executor_;
  BatchExecutor &gpu_executor_;
  const size_t total_states_;
  const size_t gpu_batch_threshold_;
  deque<shared_ptr<IntersectionWork>> intersection_queue_;
  deque<shared_ptr<SplitWork>> plane_queue_;
  deque<shared_ptr<HausdorffWork>> hausdorff_queue_;
  deque<shared_ptr<ComponentWork>> component_queue_;
  deque<shared_ptr<HullWork>> hull_queue_;
  deque<shared_ptr<FlatSurfaceWork>> flat_surface_queue_;
  deque<shared_ptr<MergeWork>> merge_queue_;
  size_t intersection_request_count_ = 0;
  size_t hausdorff_request_count_ = 0;
  size_t component_request_count_ = 0;
  size_t hull_request_count_ = 0;
  size_t flat_surface_request_count_ = 0;
  size_t merge_request_count_ = 0;
  size_t active_tasks_ = 0;
  size_t completed_states_ = 0;
  size_t next_gpu_kind_ = 0;
  array<array<bool, 2>, kGpuWorkKindCount> gpu_busy_{};
  array<bool, kGpuWorkKindCount> gpu_split_pending_{};
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

void initialize_state(Mesh mesh, size_t mesh_index, BatchState &state,
                      StageProfiler &profiler) {
  state.original_bbox = mesh.normalize();
  log(mesh_index, "Preprocessing mesh (" + to_string(mesh.vertices.size()) +
                      " verts)...");

  Mesh original_mesh = mesh.copy();
  profiled_manifold_preprocess(mesh, 30, 0.55 / 30, profiler);
  if (mesh.vertices.size() > 15000) {
    mesh = original_mesh.copy();
    profiled_manifold_preprocess(mesh, 20, 0.55 / 20, profiler);
  }
  log(mesh_index,
      "Remeshed to " + to_string(mesh.vertices.size()) + " verts.");

  state.cage = mesh.copy();
  profiled_manifold_preprocess(state.cage, 40, 0.03, profiler);

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

void finish_candidate_sampling(SplitWork &work, RandomEngine &engine) {
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

bool gpu_edge_projection_enabled() {
  const char *disabled = getenv("VISACD_DISABLE_GPU_EDGE_PROJECTION");
  return !disabled || !*disabled || string(disabled) == "0";
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

size_t configured_gpu_host_threads(size_t cpu_threads) {
  return max<size_t>(
      1, min(cpu_threads, kGpuWorkKindCount));
}

string optix_configuration_key() {
  const char *preference = getenv("VISACD_OPTIX_BUILD_PREFERENCE");
  const char *concurrency = getenv("VISACD_OPTIX_MAX_CONCURRENCY");
  return string(preference && *preference ? preference : "trace") + ":" +
         string(concurrency && *concurrency ? concurrency : "automatic");
}

struct PersistentBatchResources {
  unique_ptr<BatchExecutor> executor;
  unique_ptr<BatchExecutor> gpu_executor;
  size_t executor_threads = 0;
  size_t gpu_executor_threads = 0;

  unique_ptr<OptixRuntime> optix;
  string optix_key;
  array<CandidatePlaneRuntime, 2> candidate_planes;
  array<HausdorffRuntime, 2> hausdorff;
  DeviceMeshRuntime device_meshes;
  array<ComponentBatchRuntime, 2> component_batch;
  array<ConvexHullBatchRuntime, 2> convex_hulls;
  array<FlatSurfaceBatchRuntime, 2> flat_surfaces;
  array<MergeBatchRuntime, 2> merges;

  void configure_executors(size_t cpu_threads, size_t gpu_threads) {
    if (!executor || executor_threads != cpu_threads) {
      auto replacement = make_unique<BatchExecutor>(cpu_threads);
      executor = move(replacement);
      executor_threads = cpu_threads;
    }
    if (!gpu_executor || gpu_executor_threads != gpu_threads) {
      auto replacement = make_unique<BatchExecutor>(gpu_threads);
      gpu_executor = move(replacement);
      gpu_executor_threads = gpu_threads;
    }
  }

  OptixRuntime &configured_optix() {
    const string requested_key = optix_configuration_key();
    if (!optix || optix_key != requested_key) {
      auto replacement = make_unique<OptixRuntime>();
      optix = move(replacement);
      optix_key = requested_key;
    }
    return *optix;
  }
};

PersistentBatchResources &persistent_batch_resources() {
  // Runtime state is local to the calling thread so C++ callers can invoke
  // independent batches concurrently without sharing streams or scratch
  // buffers. Repeated calls on a thread retain executors, CUDA streams,
  // pinned buffers, and device allocation capacity.
  thread_local PersistentBatchResources resources;
  return resources;
}

} // namespace

vector<ProcessResult> process_batch(MeshList meshes, double concavity,
                                    int num_parts) {
  validate_parameters(meshes, concavity, num_parts);
  if (meshes.empty())
    return {};

  StageProfiler profiler;
  const auto process_start = chrono::steady_clock::now();

  const size_t cpu_threads = configured_cpu_threads(meshes.size());
  const size_t gpu_host_threads =
      configured_gpu_host_threads(cpu_threads);
  PersistentBatchResources &resources = persistent_batch_resources();
  resources.configure_executors(cpu_threads, gpu_host_threads);
  BatchExecutor &executor = *resources.executor;
  BatchExecutor &gpu_executor = *resources.gpu_executor;
  vector<BatchState> states(meshes.size());
  vector<ProcessResult> results(meshes.size());
  PipelineCoordinator coordinator(
      executor, gpu_executor, states.size(),
      configured_gpu_batch_threshold(cpu_threads));
  const double gpu_memory_fraction =
      config.batch_memory_fraction /
      static_cast<double>(gpu_host_threads);
  OptixRuntime &optix = resources.configured_optix();
  auto &candidate_planes = resources.candidate_planes;
  auto &hausdorff = resources.hausdorff;
  DeviceMeshRuntime &device_meshes = resources.device_meshes;
  auto &component_batch = resources.component_batch;
  auto &convex_hulls = resources.convex_hulls;
  auto &flat_surfaces = resources.flat_surfaces;
  auto &merges = resources.merges;

  const uint32_t batch_seed = random_engine();
  for (size_t i = 0; i < states.size(); ++i) {
    seed_seq seed{batch_seed, static_cast<uint32_t>(i),
                  static_cast<uint32_t>(i >> 32)};
    states[i].random_engine.seed(seed);
  }

  function<void(shared_ptr<FinalizeWork>, double)> finish_state;
  function<void(shared_ptr<FinalizeWork>, double)>
      schedule_merge_or_finish;
  function<void(shared_ptr<FinalizeWork>)> schedule_final_concavity;
  function<void(size_t)> schedule_finalize;
  function<void(size_t)> schedule_split;
  function<void(size_t)> schedule_advance;
  function<void(shared_ptr<ComponentWork>)> schedule_components;
  function<void(shared_ptr<SplitWork>)> schedule_clip;
  function<void(size_t)> schedule_initial_components;

  finish_state = [&](shared_ptr<FinalizeWork> work, double final_concavity) {
    coordinator.submit_cpu(
        [&, work = move(work), final_concavity]() mutable {
          BatchState &state = states[work->state_index];
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

  schedule_merge_or_finish =
      [&](shared_ptr<FinalizeWork> finalize_work,
          double final_concavity) {
        if (!config.use_merging || finalize_work->hulls.size() < 2) {
          finish_state(move(finalize_work), final_concavity);
          return;
        }

        BatchState &state = states[finalize_work->state_index];
        if (state.parts.size() != state.part_cache.size() ||
            state.parts.size() != finalize_work->hulls.size() ||
            state.parts.size() != finalize_work->part_hausdorff.size()) {
          throw logic_error("Merge state is out of sync");
        }
        auto work = make_shared<MergeWork>();
        work->state_index = finalize_work->state_index;
        work->final_concavity = final_concavity;
        work->initial_hull_count = finalize_work->hulls.size();
        work->part_devices.reserve(state.parts.size());
        for (size_t part_index = 0; part_index < state.parts.size();
             ++part_index) {
          PartCache &cache = state.part_cache[part_index];
          if (!cache.device) {
            cache.device = device_meshes.try_upload(
                state.parts[part_index], config.batch_memory_fraction);
          }
          work->part_devices.push_back(cache.device);
        }
        work->finalize_work = move(finalize_work);
        coordinator.enqueue_merge(move(work));
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
    work->part_hausdorff.resize(state.parts.size());
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
            StageTimer timer(profiler, ProfileStage::final_hulls_cpu, 1);
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
              hull_work->estimated_size = saturating_add(
                  hull_work->estimated_size,
                  mesh_work_size(state.parts[part_index]));
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

  schedule_initial_components = [&](size_t state_index) {
    BatchState &state = states[state_index];
    auto initial = make_shared<ComponentWork>();
    initial->state_index = state_index;
    initial->initial = true;
    initial->parts = move(state.parts);
    initial->separated.resize(initial->parts.size());
    coordinator.enqueue_components(move(initial));
  };

  schedule_components = [&](shared_ptr<ComponentWork> work) {
    coordinator.submit_cpu(
        [&, work = move(work)]() mutable {
          BatchState &state = states[work->state_index];
          {
            StageTimer timer(profiler, ProfileStage::components_cpu,
                             work->parts.size());
            size_t component_count = 0;
            for (const MeshList &components : work->separated)
              component_count += components.size();
            MeshList separated;
            separated.reserve(component_count);
            for (MeshList &components : work->separated) {
              for (Mesh &part : components)
                separated.push_back(move(part));
            }
            work->parts = move(separated);
            work->separated.clear();
          }

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
                  {
                    StageTimer timer(profiler, ProfileStage::child_cage, 1);
                    profiled_manifold_preprocess(
                        intersections->pending_parts[part_index].cage, 40,
                        0.02, profiler);
                  }
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
          StageTimer timer(profiler, ProfileStage::clip_construct, 1);
          BatchState &state = states[split->state_index];
          int *first_map = nullptr;
          int *second_map = nullptr;
          MeshList new_parts;
          if (split->prepared_clip.empty()) {
            new_parts = clip(split->part, split->selected_plane, first_map,
                             second_map);
          } else {
            new_parts = clip_prepared(
                split->part, split->selected_plane, first_map, second_map,
                split->prepared_clip);
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

          auto components = make_shared<ComponentWork>();
          components->state_index = split->state_index;
          if (gpu_edge_projection_enabled() && split->part_cache.device) {
            components->projected_edge_source = split->part_cache.device;
            components->projected_vertex_maps.resize(2);
            components->projected_vertex_maps[0].assign(
                first_map, first_map + split->part.vertices.size());
            components->projected_vertex_maps[1].assign(
                second_map, second_map + split->part.vertices.size());
          } else {
            propagate_existing_edges(split->part, first_map, second_map,
                                     new_parts);
          }
          components->parts = move(new_parts);
          components->separated.resize(components->parts.size());
          delete[] first_map;
          delete[] second_map;
          coordinator.enqueue_components(move(components));
        },
        true);
  };

  for (size_t state_index = 0; state_index < states.size(); ++state_index) {
    coordinator.submit_cpu([&, state_index]() {
      {
        StageTimer timer(profiler, ProfileStage::preprocess, 1);
        initialize_state(move(meshes[state_index]), state_index,
                         states[state_index], profiler);
      }
      if (config.use_flat_surfaces) {
        auto surfaces = make_shared<FlatSurfaceWork>();
        surfaces->state_index = state_index;
        surfaces->estimated_size = mesh_work_size(states[state_index].cage);
        coordinator.enqueue_flat_surfaces(move(surfaces));
      } else {
        schedule_initial_components(state_index);
      }
    });
  }

  try {
    while (true) {
      GpuWork gpu_work = coordinator.wait_for_gpu_work();
      if (gpu_work.kind == GpuWorkKind::complete)
        break;
      const size_t gpu_lane = gpu_work.lane;
      const double work_memory_fraction =
          gpu_work.shared_memory_budget
              ? gpu_memory_fraction * 0.5
              : gpu_memory_fraction;

      if (gpu_work.kind == GpuWorkKind::flat_surfaces) {
        vector<FlatSurfaceBatchInput> inputs;
        inputs.reserve(gpu_work.flat_surfaces.size());
        for (shared_ptr<FlatSurfaceWork> &work :
             gpu_work.flat_surfaces) {
          BatchState &state = states[work->state_index];
          if (state.parts.size() != 1) {
            throw logic_error(
                "Flat-surface state must contain its initial mesh");
          }
          inputs.push_back({&state.parts.front(),
                            config.flat_surface_min_area,
                            &work->surfaces});
        }
        coordinator.submit_gpu(
            [&, lane = gpu_lane,
              memory_fraction = work_memory_fraction,
              inputs = move(inputs),
              works = move(gpu_work.flat_surfaces)]() mutable {
              try {
                {
                  StageTimer timer(profiler, ProfileStage::flat_surfaces,
                                   inputs.size());
                  extract_flat_surfaces_batch(
                      inputs, flat_surfaces[lane], configured_batch_size(),
                      memory_fraction, &executor);
                }
                for (shared_ptr<FlatSurfaceWork> &work : works) {
                  BatchState &state = states[work->state_index];
                  state.flat_surface_planes.reserve(
                      work->surfaces.size());
                  for (const Surface &surface : work->surfaces)
                    state.flat_surface_planes.push_back(surface.plane);
                  log(work->state_index,
                      "Detected " +
                          to_string(work->surfaces.size()) +
                          " flat surface(s).");
                  schedule_initial_components(work->state_index);
                }
                coordinator.complete_gpu_batch(
                    GpuWorkKind::flat_surfaces, lane);
              } catch (...) {
                coordinator.complete_gpu_batch(
                    GpuWorkKind::flat_surfaces, lane);
                throw;
              }
            });
        continue;
      }

      if (gpu_work.kind == GpuWorkKind::merges) {
        vector<MergeBatchInput> inputs;
        inputs.reserve(gpu_work.merges.size());
        for (shared_ptr<MergeWork> &work : gpu_work.merges) {
          BatchState &state = states[work->state_index];
          if (!work->finalize_work)
            throw logic_error("Merge work has no final hulls");
          log(work->state_index,
              "Evaluating GPU merge costs for " +
                  to_string(work->initial_hull_count) + " hulls...");
          inputs.push_back(
              {&state.parts, &work->finalize_work->hulls,
               &work->part_devices,
               &work->finalize_work->hull_devices,
               &work->finalize_work->part_hausdorff,
               work->final_concavity, concavity,
               &state.random_engine});
        }
        coordinator.submit_gpu(
            [&, lane = gpu_lane,
             memory_fraction = work_memory_fraction,
             inputs = move(inputs),
             works = move(gpu_work.merges)]() mutable {
              try {
                {
                  StageTimer timer(profiler, ProfileStage::merges,
                                   inputs.size());
                  merge_convex_hulls_batch(
                      inputs, merges[lane], configured_batch_size(),
                      memory_fraction, &executor);
                }
                for (shared_ptr<MergeWork> &work : works) {
                  const size_t remaining =
                      work->finalize_work->hulls.size();
                  log(work->state_index,
                      "Merged " +
                          to_string(work->initial_hull_count - remaining) +
                          " hull(s); " + to_string(remaining) +
                          " remain.");
                  finish_state(move(work->finalize_work),
                               work->final_concavity);
                }
                coordinator.complete_gpu_batch(GpuWorkKind::merges,
                                               lane);
              } catch (...) {
                coordinator.complete_gpu_batch(GpuWorkKind::merges,
                                               lane);
                throw;
              }
            });
        continue;
      }

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

        coordinator.submit_gpu(
            [&, lane = gpu_lane,
             memory_fraction = work_memory_fraction,
             requests = move(requests),
             works = move(gpu_work.intersections)]() mutable {
              try {
                vector<vector<pair<unsigned int, unsigned int>>> edges;
                {
                  StageTimer timer(profiler, ProfileStage::intersections,
                                   requests.size());
                  edges = compute_intersection_matrices(
                      requests, optix, configured_batch_size(),
                      memory_fraction, &executor);
                }
                append_intersections(requests, edges, executor);

                for (shared_ptr<IntersectionWork> &work : works) {
                  BatchState &state = states[work->state_index];
                  if (work->initial) {
                    log(work->state_index,
                        "Starting decomposition (max parts=" +
                            to_string(num_parts) +
                            ", concavity threshold=" +
                            to_string(concavity) + ", mode=" +
                            config.score_mode + ").");
                  } else {
                    for (PendingPart &pending : work->pending_parts) {
                      state.parts.push_back(move(pending.part));
                      state.part_cache.emplace_back();
                    }
                    ++state.iteration;
                  }
                  schedule_advance(work->state_index);
                }
                coordinator.complete_gpu_batch(
                    GpuWorkKind::intersections, lane);
              } catch (...) {
                coordinator.complete_gpu_batch(
                    GpuWorkKind::intersections, lane);
                throw;
              }
            });
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
        coordinator.submit_gpu(
            [&, lane = gpu_lane,
             memory_fraction = work_memory_fraction,
             inputs = move(inputs),
             works = move(gpu_work.hulls)]() mutable {
              try {
                {
                  StageTimer timer(profiler, ProfileStage::selection_hulls,
                                   inputs.size());
                  compute_convex_hulls_batch(
                      inputs, convex_hulls[lane], configured_batch_size(),
                      memory_fraction);
                }
                for (shared_ptr<HullWork> &work : works)
                  schedule_advance(work->state_index);
                coordinator.complete_gpu_batch(GpuWorkKind::hulls,
                                               lane);
              } catch (...) {
                coordinator.complete_gpu_batch(GpuWorkKind::hulls,
                                               lane);
                throw;
              }
            });
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
        coordinator.submit_gpu(
            [&, lane = gpu_lane,
             memory_fraction = work_memory_fraction,
             jobs = move(jobs),
             works = move(gpu_work.hausdorff)]() mutable {
              try {
                {
                  StageTimer timer(profiler, ProfileStage::hausdorff,
                                   jobs.size());
                  evaluate_hausdorff_batch(
                      jobs, hausdorff[lane], configured_batch_size(),
                      memory_fraction);
                }

                for (shared_ptr<HausdorffWork> &work : works) {
                  if (work->purpose == HausdorffPurpose::selection) {
                    if (work->part_indices.size() != work->jobs.size() ||
                        work->relative_volume_terms.size() !=
                            work->jobs.size()) {
                      throw logic_error(
                          "Selection Hausdorff work is out of sync");
                    }
                    BatchState &state = states[work->state_index];
                    for (size_t job_index = 0;
                         job_index < work->jobs.size(); ++job_index) {
                      const size_t part_index =
                          work->part_indices[job_index];
                      if (part_index >= state.part_cache.size()) {
                        throw logic_error(
                            "Selection part index is out of range");
                      }
                      state.part_cache[part_index]
                          .selection_concavity =
                          max(work->relative_volume_terms[job_index],
                              work->jobs[job_index].result);
                    }
                    schedule_split(work->state_index);
                    continue;
                  }

                  if (!work->finalize_work) {
                    throw logic_error(
                        "Final Hausdorff work has no hulls");
                  }
                  double final_concavity = 0.0;
                  for (size_t job_index = 0;
                       job_index < work->jobs.size(); ++job_index) {
                    const double result = work->jobs[job_index].result;
                    work->finalize_work
                        ->part_hausdorff[job_index] = result;
                    final_concavity = max(final_concavity, result);
                  }
                  schedule_merge_or_finish(
                      move(work->finalize_work), final_concavity);
                }
                coordinator.complete_gpu_batch(
                    GpuWorkKind::hausdorff, lane);
              } catch (...) {
                coordinator.complete_gpu_batch(
                    GpuWorkKind::hausdorff, lane);
                throw;
              }
            });
        continue;
      }

      if (gpu_work.kind == GpuWorkKind::components) {
        size_t input_count = 0;
        for (const shared_ptr<ComponentWork> &work : gpu_work.components)
          input_count += work->parts.size();
        vector<ComponentBatchInput> inputs;
        inputs.reserve(input_count);
        for (shared_ptr<ComponentWork> &work : gpu_work.components) {
          if (work->separated.size() != work->parts.size())
            throw logic_error("Component work is out of sync");
          for (size_t part_index = 0; part_index < work->parts.size();
               ++part_index) {
            const bool projects_edges =
                work->projected_edge_source &&
                part_index < work->projected_vertex_maps.size();
            inputs.push_back(
                {&work->parts[part_index], nullptr,
                 &work->separated[part_index],
                 projects_edges ? work->projected_edge_source.get()
                                : nullptr,
                 projects_edges
                     ? &work->projected_vertex_maps[part_index]
                     : nullptr});
          }
        }
        coordinator.submit_gpu(
            [&, lane = gpu_lane,
             memory_fraction = work_memory_fraction,
             inputs = move(inputs),
             works = move(gpu_work.components)]() mutable {
              try {
                {
                  StageTimer timer(profiler, ProfileStage::components_gpu,
                                   inputs.size());
                  separate_components_batch(
                      inputs, component_batch[lane],
                      configured_batch_size(), memory_fraction);
                }
                for (shared_ptr<ComponentWork> &work : works)
                  schedule_components(move(work));
                coordinator.complete_gpu_batch(
                    GpuWorkKind::components, lane);
              } catch (...) {
                coordinator.complete_gpu_batch(
                    GpuWorkKind::components, lane);
                throw;
              }
            });
        continue;
      }

      vector<SplitPlaneInput> candidate_inputs;
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
        BatchState &state = states[split->state_index];
        candidate_inputs.push_back(
            {&split->part, split->part_cache.device.get(),
             split->sampled_edges.data(), split->sampled_edges.size(),
             kCandidatePlaneCount, &state.flat_surface_planes,
             static_cast<float>(config.flat_surface_k),
             &split->selected_plane, &split->has_selected_plane,
             &split->prepared_clip,
             &split->candidate_attempts});
      }
      coordinator.submit_gpu(
          [&, lane = gpu_lane,
           memory_fraction = work_memory_fraction,
           candidate_inputs = move(candidate_inputs),
           splits = move(gpu_work.planes)]() mutable {
            try {
              {
                StageTimer timer(profiler, ProfileStage::split_fused,
                                 candidate_inputs.size());
                generate_score_select_clip_batch(
                    candidate_inputs, candidate_planes[lane],
                    configured_batch_size(), memory_fraction);
              }

              for (shared_ptr<SplitWork> &split : splits) {
                BatchState &state = states[split->state_index];
                finish_candidate_sampling(*split, state.random_engine);
                if (!split->has_selected_plane) {
                  state.parts.push_back(move(split->part));
                  state.part_cache.push_back(move(split->part_cache));
                  state.finished = true;
                  schedule_finalize(split->state_index);
                  continue;
                }
                schedule_clip(move(split));
              }
              coordinator.complete_gpu_batch(GpuWorkKind::planes,
                                             lane);
            } catch (...) {
              coordinator.complete_gpu_batch(GpuWorkKind::planes,
                                             lane);
              throw;
            }
          });
    }
  } catch (...) {
    coordinator.set_error(current_exception());
    coordinator.wait_for_gpu_work();
    throw;
  }

  profiler.report(chrono::steady_clock::now() - process_start,
                  states.size());
  return results;
}

} // namespace neural_acd
