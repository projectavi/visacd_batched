#include <algorithm>
#include <atomic>
#include <clip.hpp>
#include <condition_variable>
#include <config.hpp>
#include <core.hpp>
#include <cost.hpp>
#include <exception>
#include <functional>
#include <iomanip>
#include <intersections.hpp>
#include <iostream>
#include <limits>
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

struct BatchState {
  Mesh cage;
  MeshList parts;
  vector<double> original_bbox;
  vector<Plane> flat_surface_planes;
  RandomEngine random_engine;
  bool finished = false;
};

struct SplitWork {
  size_t state_index;
  Mesh part;
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

class BatchExecutor {
public:
  explicit BatchExecutor(size_t thread_count)
      : thread_count_(max<size_t>(1, thread_count)) {
    workers_.reserve(thread_count_ - 1);
    for (size_t i = 1; i < thread_count_; ++i)
      workers_.emplace_back([this]() { worker_loop(); });
  }

  ~BatchExecutor() {
    {
      lock_guard<mutex> lock(job_mutex_);
      stopping_ = true;
    }
    start_condition_.notify_all();
    for (thread &worker : workers_)
      worker.join();
  }

  BatchExecutor(const BatchExecutor &) = delete;
  BatchExecutor &operator=(const BatchExecutor &) = delete;

  template <typename Function>
  void parallel_for(size_t work_size, Function function) {
    if (work_size == 0)
      return;
    if (thread_count_ <= 1) {
      for (size_t i = 0; i < work_size; ++i)
        function(i);
      return;
    }

    {
      lock_guard<mutex> lock(job_mutex_);
      function_ = move(function);
      work_size_ = work_size;
      next_.store(0, memory_order_relaxed);
      failed_.store(false, memory_order_relaxed);
      exception_ = nullptr;
      workers_remaining_ = workers_.size();
      ++generation_;
    }
    start_condition_.notify_all();

    run_job();

    unique_lock<mutex> lock(job_mutex_);
    done_condition_.wait(lock,
                         [this]() { return workers_remaining_ == 0; });
    exception_ptr exception = exception_;
    function_ = nullptr;
    lock.unlock();
    if (exception)
      rethrow_exception(exception);
  }

private:
  void run_job() {
    while (!failed_.load(memory_order_relaxed)) {
      const size_t i = next_.fetch_add(1, memory_order_relaxed);
      if (i >= work_size_)
        return;
      try {
        function_(i);
      } catch (...) {
        bool expected = false;
        if (failed_.compare_exchange_strong(expected, true,
                                            memory_order_relaxed)) {
          lock_guard<mutex> lock(exception_mutex_);
          exception_ = current_exception();
        }
      }
    }
  }

  void worker_loop() {
    size_t observed_generation = 0;
    unique_lock<mutex> lock(job_mutex_);
    while (true) {
      start_condition_.wait(lock, [this, observed_generation]() {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_)
        return;
      observed_generation = generation_;
      lock.unlock();
      run_job();
      lock.lock();
      if (--workers_remaining_ == 0)
        done_condition_.notify_one();
    }
  }

  size_t thread_count_;
  vector<thread> workers_;
  mutex job_mutex_;
  mutex exception_mutex_;
  condition_variable start_condition_;
  condition_variable done_condition_;
  function<void(size_t)> function_;
  size_t work_size_ = 0;
  atomic<size_t> next_{0};
  atomic<bool> failed_{false};
  exception_ptr exception_;
  size_t workers_remaining_ = 0;
  size_t generation_ = 0;
  bool stopping_ = false;
};

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
}

SplitWork make_split_work(size_t state_index, Mesh part,
                          const vector<Plane> &flat_surface_planes,
                          RandomEngine &engine) {
  SplitWork work;
  work.state_index = state_index;
  work.part = move(part);
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
    const vector<vector<pair<unsigned int, unsigned int>>> &edges) {
  for (size_t i = 0; i < requests.size(); ++i) {
    auto &destination = requests[i].first->intersecting_edges;
    destination.insert(destination.end(), edges[i].begin(), edges[i].end());
  }
}

size_t configured_batch_size() {
  return config.max_batch_size > 0
             ? static_cast<size_t>(config.max_batch_size)
             : 0;
}

} // namespace

vector<ProcessResult> process_batch(MeshList meshes, double concavity,
                                    int num_parts) {
  validate_parameters(meshes, concavity, num_parts);
  if (meshes.empty())
    return {};

  BatchExecutor executor(configured_cpu_threads(meshes.size()));
  vector<BatchState> states(meshes.size());
  const uint32_t batch_seed = random_engine();
  for (size_t i = 0; i < states.size(); ++i) {
    seed_seq seed{batch_seed, static_cast<uint32_t>(i),
                  static_cast<uint32_t>(i >> 32)};
    states[i].random_engine.seed(seed);
  }
  executor.parallel_for(meshes.size(), [&](size_t i) {
    initialize_state(move(meshes[i]), i, states[i]);
  });

  OptixRuntime optix;
  vector<pair<Mesh *, Mesh *>> initial_requests;
  for (BatchState &state : states) {
    for (Mesh &part : state.parts)
      initial_requests.emplace_back(&part, &state.cage);
  }
  auto initial_edges = compute_intersection_matrices(
      initial_requests, optix, configured_batch_size(),
      config.batch_memory_fraction);
  append_intersections(initial_requests, initial_edges);

  for (size_t i = 0; i < states.size(); ++i) {
    log(i, "Starting decomposition (max parts=" + to_string(num_parts) +
               ", concavity threshold=" + to_string(concavity) +
               ", mode=" + config.score_mode + ").");
  }

  for (int iteration = 0; iteration < num_parts - 1; ++iteration) {
    vector<optional<SplitWork>> split_by_state(states.size());
    executor.parallel_for(states.size(), [&](size_t state_index) {
      BatchState &state = states[state_index];
      if (state.finished)
        return;

      int part_index = -1;
      if (config.score_mode == "edge") {
        part_index = get_part_with_highest_score(state.parts);
      } else if (config.score_mode == "concavity") {
        double max_concavity = -1.0;
        part_index = get_part_with_highest_concavity(
            state.parts, max_concavity, state.random_engine);
        if (max_concavity < concavity) {
          log(state_index,
              "Concavity " + to_string(max_concavity) +
                  " is below threshold; stopping.");
          state.finished = true;
          return;
        }
      }

      if (part_index < 0) {
        state.finished = true;
        return;
      }

      log(state_index,
          "Step " + to_string(iteration + 1) + "/" +
              to_string(num_parts - 1) + ": splitting part " +
              to_string(part_index) + " (" +
              to_string(state.parts.size()) + " parts total).");

      Mesh part = move(state.parts[part_index]);
      state.parts.erase(state.parts.begin() + part_index);
      if (part.intersecting_edges.empty()) {
        state.parts.push_back(move(part));
        state.finished = true;
        log(state_index, "No more intersecting edges; stopping early.");
        return;
      }

      SplitWork split = make_split_work(
          state_index, move(part), state.flat_surface_planes,
          state.random_engine);
      if (split.planes.empty()) {
        state.parts.push_back(move(split.part));
        state.finished = true;
        return;
      }
      split_by_state[state_index].emplace(move(split));
    });

    vector<SplitWork> work;
    work.reserve(states.size());
    for (optional<SplitWork> &split : split_by_state) {
      if (split)
        work.push_back(move(*split));
    }

    vector<PlaneScoreInput> score_inputs;
    score_inputs.reserve(work.size());
    for (SplitWork &split : work) {
      if (split.planes.size() >
              static_cast<size_t>(numeric_limits<int>::max()) ||
          split.part.vertices.size() >
              static_cast<size_t>(numeric_limits<int>::max()) ||
          split.part.intersecting_edges.size() >
              static_cast<size_t>(numeric_limits<int>::max())) {
        throw overflow_error("Plane scoring input is too large");
      }
      score_inputs.push_back(
          {split.host_planes.data(), split.host_points.data(),
           split.host_edges.data(), split.scores.data(),
           static_cast<int>(split.planes.size()),
           static_cast<int>(split.part.vertices.size()),
           static_cast<int>(split.part.intersecting_edges.size())});
    }
    classify_and_rate_planes_batch(score_inputs, configured_batch_size(),
                                   config.batch_memory_fraction);

    vector<vector<PendingPart>> pending_by_work(work.size());
    executor.parallel_for(work.size(), [&](size_t work_index) {
      SplitWork &split = work[work_index];
      for (size_t i = split.flat_surface_offset; i < split.scores.size(); ++i)
        split.scores[i] *= static_cast<float>(config.flat_surface_k);

      const size_t best_index =
          max_element(split.scores.begin(), split.scores.end()) -
          split.scores.begin();
      int *first_map = nullptr;
      int *second_map = nullptr;
      MeshList new_parts =
          clip(split.part, split.planes[best_index], first_map, second_map);
      if (new_parts.size() < 2) {
        delete[] first_map;
        delete[] second_map;
        states[split.state_index].parts.push_back(move(split.part));
        return;
      }

      propagate_existing_edges(split.part, first_map, second_map, new_parts);
      delete[] first_map;
      delete[] second_map;
      separate_disjoint(new_parts);

      for (Mesh &part : new_parts) {
        if (part.vertices.size() < 10)
          continue;
        PendingPart pending;
        pending.state_index = split.state_index;
        pending.part = move(part);
        pending.cage = pending.part.copy();
        manifold_preprocess(pending.cage, 40, 0.02);
        pending_by_work[work_index].push_back(move(pending));
      }
    });

    vector<PendingPart> pending_parts;
    for (vector<PendingPart> &pending_group : pending_by_work) {
      for (PendingPart &pending : pending_group)
        pending_parts.push_back(move(pending));
    }

    vector<pair<Mesh *, Mesh *>> requests;
    requests.reserve(pending_parts.size());
    for (PendingPart &pending : pending_parts)
      requests.emplace_back(&pending.part, &pending.cage);

    auto new_edges = compute_intersection_matrices(
        requests, optix, configured_batch_size(),
        config.batch_memory_fraction);
    append_intersections(requests, new_edges);
    for (PendingPart &pending : pending_parts) {
      states[pending.state_index].parts.push_back(move(pending.part));
    }
  }

  vector<MeshList> hulls_by_state(states.size());
  vector<double> final_concavities(states.size());
  executor.parallel_for(states.size(), [&](size_t state_index) {
    BatchState &state = states[state_index];
    log(state_index, "Computing convex hulls for " +
                         to_string(state.parts.size()) + " parts...");

    MeshList &hulls = hulls_by_state[state_index];
    hulls.reserve(state.parts.size());
    for (Mesh &part : state.parts) {
      Mesh hull;
      part.compute_ch(hull, true);
      hulls.push_back(move(hull));
    }

    final_concavities[state_index] = compute_final_concavity(
        state.parts, hulls, state.random_engine);
  });

  vector<ProcessResult> results;
  results.reserve(states.size());
  for (size_t state_index = 0; state_index < states.size(); ++state_index) {
    BatchState &state = states[state_index];
    MeshList &hulls = hulls_by_state[state_index];
    const double final_concavity = final_concavities[state_index];
    if (config.use_merging)
      multimerge_ch(state.parts, hulls, final_concavity, concavity);

    for (Mesh &hull : hulls)
      hull.unnormalize(state.original_bbox);
    for (Mesh &part : state.parts)
      part.unnormalize(state.original_bbox);

    MeshList output = config.return_parts ? move(state.parts) : move(hulls);
    const int output_count = static_cast<int>(output.size());
    ostringstream summary;
    summary << "Done. parts=" << output_count
            << "  concavity=" << fixed << setprecision(4)
            << final_concavity;
    log(state_index, summary.str());
    results.push_back(
        {move(output), final_concavity, output_count});
  }

  return results;
}

} // namespace neural_acd
