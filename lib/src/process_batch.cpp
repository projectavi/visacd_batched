#include <algorithm>
#include <atomic>
#include <batch_executor.hpp>
#include <chrono>
#include <clip.hpp>
#include <config.hpp>
#include <condition_variable>
#include <core.hpp>
#include <cost.hpp>
#include <deque>
#include <exception>
#include <functional>
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
double compute_final_concavity(MeshList &parts, MeshList &hulls,
                               RandomEngine &engine);

namespace {

constexpr size_t kMaxAutomaticCpuThreads = 200;

struct PartCache {
  optional<Mesh> hull;
  optional<double> selection_concavity;
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
  vector<float> host_planes;
  vector<float> host_points;
  vector<unsigned int> host_edges;
  vector<float> scores;
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

struct FinalizeWork {
  size_t state_index;
  MeshList hulls;
  atomic<size_t> remaining{0};
};

enum class GpuWorkKind { intersections, planes, complete };

struct GpuWork {
  GpuWorkKind kind;
  vector<shared_ptr<IntersectionWork>> intersections;
  vector<shared_ptr<SplitWork>> planes;
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
               !plane_queue_.empty();
      });

      if (error_) {
        if (active_cpu_tasks_ != 0) {
          condition_.wait(lock, [this]() { return active_cpu_tasks_ == 0; });
        }
        rethrow_exception(error_);
      }
      if (completed())
        return {GpuWorkKind::complete, {}, {}};

      if (!gpu_batch_ready() && active_cpu_tasks_ != 0) {
        condition_.wait_for(lock, chrono::milliseconds(1), [this]() {
          return error_ || completed() || gpu_batch_ready() ||
                 active_cpu_tasks_ == 0;
        });
        if (error_)
          continue;
        if (completed())
          return {GpuWorkKind::complete, {}, {}};
      }

      const bool have_intersections = !intersection_queue_.empty();
      const bool have_planes = !plane_queue_.empty();
      bool take_intersections = have_intersections;
      if (have_intersections && have_planes) {
        take_intersections = prefer_intersections_;
        prefer_intersections_ = !prefer_intersections_;
      }

      if (take_intersections) {
        GpuWork result{GpuWorkKind::intersections, {}, {}};
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

      GpuWork result{GpuWorkKind::planes, {}, {}};
      result.planes.reserve(plane_queue_.size());
      while (!plane_queue_.empty()) {
        result.planes.push_back(move(plane_queue_.front()));
        plane_queue_.pop_front();
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
           plane_queue_.size() >= gpu_batch_threshold_;
  }

  void record_error(exception_ptr error) {
    if (!error_)
      error_ = error;
    intersection_queue_.clear();
    plane_queue_.clear();
    intersection_request_count_ = 0;
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
  size_t intersection_request_count_ = 0;
  size_t active_cpu_tasks_ = 0;
  size_t completed_states_ = 0;
  bool prefer_intersections_ = true;
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
  separate_disjoint(state.parts);
  state.part_cache.resize(state.parts.size());
}

SplitWork make_split_work(size_t state_index, Mesh part, PartCache part_cache,
                          const vector<Plane> &flat_surface_planes,
                          RandomEngine &engine) {
  SplitWork work;
  work.state_index = state_index;
  work.part = move(part);
  work.part_cache = move(part_cache);
  work.planes = get_candidate_planes(work.part.vertices,
                                      work.part.intersecting_edges, 2500,
                                      engine);
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
    work.host_edges[i * 2 + 1] = work.part.intersecting_edges[i].second;
  }
  work.scores.assign(work.planes.size(), 0.0f);
  return work;
}

int get_part_with_highest_concavity_cached(BatchState &state,
                                            double &max_concavity) {
  if (state.parts.size() != state.part_cache.size())
    throw logic_error("Part cache is out of sync with batch state");

  int best_index = -1;
  for (size_t part_index = 0; part_index < state.parts.size(); ++part_index) {
    PartCache &cache = state.part_cache[part_index];
    if (!cache.hull) {
      cache.hull.emplace();
      state.parts[part_index].compute_ch(*cache.hull, true);
    }
    if (!cache.selection_concavity) {
      cache.selection_concavity =
          compute_h(state.parts[part_index], *cache.hull, 0.3, 3000, 42,
                    false, state.random_engine);
    }
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
  PlaneScoringRuntime plane_scoring;

  const uint32_t batch_seed = random_engine();
  for (size_t i = 0; i < states.size(); ++i) {
    seed_seq seed{batch_seed, static_cast<uint32_t>(i),
                  static_cast<uint32_t>(i >> 32)};
    states[i].random_engine.seed(seed);
  }

  function<void(shared_ptr<FinalizeWork>)> finish_state;
  function<void(size_t)> schedule_finalize;
  function<void(size_t)> schedule_advance;
  function<void(shared_ptr<SplitWork>)> schedule_clip;

  finish_state = [&](shared_ptr<FinalizeWork> work) {
    coordinator.submit_cpu(
        [&, work = move(work)]() mutable {
          BatchState &state = states[work->state_index];
          const double final_concavity = compute_final_concavity(
              state.parts, work->hulls, state.random_engine);
          if (config.use_merging) {
            multimerge_ch(state.parts, work->hulls, final_concavity,
                          concavity);
          }

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

  schedule_finalize = [&](size_t state_index) {
    BatchState &state = states[state_index];
    if (state.finalizing)
      return;
    state.finalizing = true;

    auto work = make_shared<FinalizeWork>();
    work->state_index = state_index;
    work->hulls.resize(state.parts.size());
    log(state_index, "Computing convex hulls for " +
                         to_string(state.parts.size()) + " parts...");

    if (state.parts.size() != state.part_cache.size())
      throw logic_error("Part cache is out of sync during finalization");

    vector<size_t> missing_hulls;
    for (size_t part_index = 0; part_index < state.parts.size(); ++part_index) {
      if (state.part_cache[part_index].hull) {
        work->hulls[part_index] =
            move(*state.part_cache[part_index].hull);
      } else {
        missing_hulls.push_back(part_index);
      }
    }
    work->remaining.store(missing_hulls.size());

    if (missing_hulls.empty()) {
      finish_state(move(work));
      return;
    }

    for (size_t part_index : missing_hulls) {
      coordinator.submit_cpu(
          [&, work, part_index]() {
            states[work->state_index].parts[part_index].compute_ch(
                work->hulls[part_index], true);
            if (work->remaining.fetch_sub(1) == 1)
              finish_state(work);
          },
          true);
    }
  };

  schedule_advance = [&](size_t state_index) {
    coordinator.submit_cpu(
        [&, state_index]() {
          BatchState &state = states[state_index];
          if (state.finished || state.iteration >= num_parts - 1) {
            schedule_finalize(state_index);
            return;
          }

          int part_index = -1;
          if (config.score_mode == "edge") {
            part_index = get_part_with_highest_score(state.parts);
          } else if (config.score_mode == "concavity") {
            double max_concavity = -1.0;
            part_index =
                get_part_with_highest_concavity_cached(state, max_concavity);
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

          auto split = make_shared<SplitWork>(make_split_work(
              state_index, move(part), move(part_cache),
              state.flat_surface_planes,
              state.random_engine));
          if (split->planes.empty()) {
            state.parts.push_back(move(split->part));
            state.part_cache.push_back(move(split->part_cache));
            state.finished = true;
            schedule_finalize(state_index);
            return;
          }
          coordinator.enqueue_planes(move(split));
        },
        true);
  };

  schedule_clip = [&](shared_ptr<SplitWork> split) {
    coordinator.submit_cpu(
        [&, split = move(split)]() mutable {
          BatchState &state = states[split->state_index];
          for (size_t i = split->flat_surface_offset;
               i < split->scores.size(); ++i) {
            split->scores[i] *= static_cast<float>(config.flat_surface_k);
          }

          const size_t best_index =
              max_element(split->scores.begin(), split->scores.end()) -
              split->scores.begin();
          int *first_map = nullptr;
          int *second_map = nullptr;
          MeshList new_parts = clip(split->part, split->planes[best_index],
                                    first_map, second_map);
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
          separate_disjoint(new_parts);

          auto intersections = make_shared<IntersectionWork>();
          intersections->state_index = split->state_index;
          for (Mesh &part : new_parts) {
            if (part.vertices.size() < 10)
              continue;
            PendingPart pending;
            pending.state_index = split->state_index;
            pending.part = move(part);
            pending.cage = pending.part.copy();
            intersections->pending_parts.push_back(move(pending));
          }

          if (intersections->pending_parts.empty()) {
            ++state.iteration;
            schedule_advance(split->state_index);
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

  for (size_t state_index = 0; state_index < states.size(); ++state_index) {
    coordinator.submit_cpu([&, state_index]() {
      initialize_state(move(meshes[state_index]), state_index,
                       states[state_index]);
      auto initial = make_shared<IntersectionWork>();
      initial->state_index = state_index;
      initial->initial = true;
      for (Mesh &part : states[state_index].parts)
        initial->requests.emplace_back(&part, &states[state_index].cage);
      coordinator.enqueue_intersections(move(initial));
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

      vector<PlaneScoreInput> score_inputs;
      score_inputs.reserve(gpu_work.planes.size());
      for (shared_ptr<SplitWork> &split : gpu_work.planes) {
        if (split->planes.size() >
                static_cast<size_t>(numeric_limits<int>::max()) ||
            split->part.vertices.size() >
                static_cast<size_t>(numeric_limits<int>::max()) ||
            split->part.intersecting_edges.size() >
                static_cast<size_t>(numeric_limits<int>::max())) {
          throw overflow_error("Plane scoring input is too large");
        }
        score_inputs.push_back(
            {split->host_planes.data(), split->host_points.data(),
             split->host_edges.data(), split->scores.data(),
             static_cast<int>(split->planes.size()),
             static_cast<int>(split->part.vertices.size()),
             static_cast<int>(split->part.intersecting_edges.size())});
      }
      classify_and_rate_planes_batch(score_inputs, plane_scoring,
                                     configured_batch_size(),
                                     config.batch_memory_fraction);
      for (shared_ptr<SplitWork> &split : gpu_work.planes)
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
