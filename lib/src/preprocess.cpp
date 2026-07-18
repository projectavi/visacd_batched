#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/util/Util.h>
#include <preprocess.hpp>
#include <preprocess_cuda.hpp>
#include <preprocess_expand_cuda.hpp>
#include <preprocess_renormalize_cuda.hpp>
#include <preprocess_surface_post_cuda.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace openvdb;
using namespace std;

namespace neural_acd {

namespace {

using PreprocessClock = chrono::steady_clock;

long long elapsed_ns(PreprocessClock::time_point start) {
  return chrono::duration_cast<chrono::nanoseconds>(
             PreprocessClock::now() - start)
      .count();
}

bool environment_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && string(value) != "0";
}

unsigned int dense_narrowband_wait_microseconds() {
  const char *value =
      std::getenv("VISACD_PREPROCESS_EXPAND_DENSE_WAIT_US");
  if (!value || !*value)
    return 250;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || *end != '\0' || parsed > 100000)
    throw invalid_argument(
        "VISACD_PREPROCESS_EXPAND_DENSE_WAIT_US must be in [0, 100000]");
  return static_cast<unsigned int>(parsed);
}

using NarrowbandTree = DoubleGrid::TreeType;
using NarrowbandLeaf = NarrowbandTree::LeafNodeType;
using NarrowbandIndexTree =
    NarrowbandTree::ValueConverter<Int32>::Type;
using NarrowbandBoolTree =
    NarrowbandTree::ValueConverter<bool>::Type;
using NarrowbandBoolLeaf = NarrowbandBoolTree::LeafNodeType;

struct NarrowbandLeafWork {
  Coord origin;
  CoordBBox bounds;
  NarrowbandLeaf::NodeMaskType second_mask;
  size_t fragment_offset = 0;
  size_t fragment_count = 0;
  size_t first_candidate_offset = 0;
  size_t first_candidate_count = 0;
  size_t second_candidate_offset = 0;
  size_t second_candidate_count = 0;
};

void gather_narrowband_fragments(
    const CoordBBox &bounds,
    tree::ValueAccessor<NarrowbandTree> &distance_accessor,
    tree::ValueAccessor<NarrowbandIndexTree> &index_accessor,
    vector<NarrowbandFragment> &fragments) {
  const Coord node_minimum =
      bounds.min() & ~(NarrowbandLeaf::DIM - 1);
  const Coord node_maximum =
      bounds.max() & ~(NarrowbandLeaf::DIM - 1);
  Coord origin;
  for (origin[0] = node_minimum[0]; origin[0] <= node_maximum[0];
       origin[0] += NarrowbandLeaf::DIM) {
    for (origin[1] = node_minimum[1]; origin[1] <= node_maximum[1];
         origin[1] += NarrowbandLeaf::DIM) {
      for (origin[2] = node_minimum[2]; origin[2] <= node_maximum[2];
           origin[2] += NarrowbandLeaf::DIM) {
        const NarrowbandLeaf *distance_leaf =
            distance_accessor.probeConstLeaf(origin);
        if (!distance_leaf)
          continue;
        const NarrowbandIndexTree::LeafNodeType *index_leaf =
            index_accessor.probeConstLeaf(origin);
        if (!index_leaf)
          throw logic_error("Narrowband distance and index trees diverged");
        const Coord region_minimum =
            Coord::maxComponent(bounds.min(), origin);
        const Coord region_maximum = Coord::minComponent(
            bounds.max(),
            origin.offsetBy(NarrowbandLeaf::DIM - 1));
        const auto &mask = distance_leaf->getValueMask();
        const Int32 *indices = index_leaf->buffer().data();
        for (int x = region_minimum[0]; x <= region_maximum[0]; ++x) {
          for (int y = region_minimum[1]; y <= region_maximum[1]; ++y) {
            for (int z = region_minimum[2]; z <= region_maximum[2]; ++z) {
              const Coord coordinate(x, y, z);
              const Index position =
                  NarrowbandLeaf::coordToOffset(coordinate);
              if (mask.isOn(position)) {
                fragments.push_back(
                    {indices[position], x, y, z});
              }
            }
          }
        }
      }
    }
  }
}

bool update_narrowband_voxel(
    const Coord &coordinate, const NarrowbandDistance &evaluation,
    tree::ValueAccessor<NarrowbandTree> &distance_accessor,
    tree::ValueAccessor<NarrowbandIndexTree> &index_accessor,
    double exterior_width, double interior_width, double voxel_size) {
  const bool inside = distance_accessor.getValue(coordinate) < 0.0;
  if (!inside && evaluation.distance < exterior_width) {
    distance_accessor.setValue(coordinate, evaluation.distance);
    index_accessor.setValue(coordinate, evaluation.triangle_index);
    return evaluation.distance + voxel_size < exterior_width;
  }
  if (inside && evaluation.distance < interior_width) {
    distance_accessor.setValue(coordinate, -evaluation.distance);
    index_accessor.setValue(coordinate, evaluation.triangle_index);
    return evaluation.distance + voxel_size < interior_width;
  }
  return false;
}

void evaluate_narrowband_distances(
    const Mesh &source_mesh, double scale, const vector<Vec3s> &points,
    const vector<Vec3I> &triangles, double voxel_size,
    const vector<NarrowbandFragment> &fragments,
    const vector<NarrowbandCandidate> &candidates,
    vector<NarrowbandDistance> &distances);

void expand_narrowband_cuda_iteration(
    NarrowbandTree &distance_tree, NarrowbandIndexTree &index_tree,
    NarrowbandBoolTree &mask_tree,
    vector<NarrowbandBoolLeaf *> &mask_nodes, const Mesh &source_mesh,
    double scale, const vector<Vec3s> &points,
    const vector<Vec3I> &triangles, double exterior_width,
    double interior_width, double voxel_size) {
  vector<NarrowbandLeafWork> work;
  vector<NarrowbandFragment> fragments;
  vector<NarrowbandCandidate> first_candidates;
  work.reserve(mask_nodes.size());
  tree::ValueAccessor<NarrowbandTree> distance_accessor(distance_tree);
  tree::ValueAccessor<NarrowbandIndexTree> index_accessor(index_tree);

  for (NarrowbandBoolLeaf *mask_leaf : mask_nodes) {
    if (!mask_leaf || mask_leaf->isEmpty())
      continue;
    NarrowbandLeafWork leaf_work;
    leaf_work.origin = mask_leaf->origin();
    leaf_work.bounds = mask_leaf->getNodeBoundingBox();
    CoordBBox fragment_bounds(Coord::max(), Coord::min());
    for (auto iterator = mask_leaf->cbeginValueOn(); iterator; ++iterator)
      fragment_bounds.expand(iterator.getCoord());
    fragment_bounds.expand(1);
    leaf_work.fragment_offset = fragments.size();
    gather_narrowband_fragments(fragment_bounds, distance_accessor,
                                index_accessor, fragments);
    leaf_work.fragment_count =
        fragments.size() - leaf_work.fragment_offset;
    sort(fragments.begin() + leaf_work.fragment_offset, fragments.end(),
         [](const NarrowbandFragment &first,
            const NarrowbandFragment &second) {
           return first.triangle_index < second.triangle_index;
         });
    leaf_work.first_candidate_offset = first_candidates.size();
    for (auto iterator = mask_leaf->cbeginValueOn(); iterator; ++iterator) {
      const Coord coordinate = iterator.getCoord();
      first_candidates.push_back(
          {coordinate[0], coordinate[1], coordinate[2], 5,
           leaf_work.fragment_offset, leaf_work.fragment_count});
    }
    leaf_work.first_candidate_count =
        first_candidates.size() - leaf_work.first_candidate_offset;
    work.push_back(move(leaf_work));
  }

  vector<NarrowbandDistance> first_distances;
  evaluate_narrowband_distances(
      source_mesh, scale, points, triangles, voxel_size, fragments,
      first_candidates, first_distances);

  NarrowbandBoolTree next_mask_tree(false);
  tree::ValueAccessor<NarrowbandBoolTree> next_mask_accessor(next_mask_tree);
  vector<NarrowbandCandidate> second_candidates;
  for (NarrowbandLeafWork &leaf_work : work) {
    for (size_t local = 0; local < leaf_work.first_candidate_count;
         ++local) {
      const size_t candidate_index =
          leaf_work.first_candidate_offset + local;
      const NarrowbandCandidate &candidate =
          first_candidates[candidate_index];
      const Coord coordinate(candidate.x, candidate.y, candidate.z);
      if (!update_narrowband_voxel(
              coordinate, first_distances[candidate_index],
              distance_accessor, index_accessor, exterior_width,
              interior_width, voxel_size)) {
        continue;
      }
      for (int neighbour = 0; neighbour < 6; ++neighbour) {
        const Coord adjacent =
            coordinate + util::COORD_OFFSETS[neighbour];
        if (leaf_work.bounds.isInside(adjacent)) {
          leaf_work.second_mask.setOn(
              NarrowbandLeaf::coordToOffset(adjacent));
        } else {
          next_mask_accessor.setValueOn(adjacent);
        }
      }
      for (int neighbour = 6; neighbour < 26; ++neighbour) {
        const Coord adjacent =
            coordinate + util::COORD_OFFSETS[neighbour];
        if (leaf_work.bounds.isInside(adjacent)) {
          leaf_work.second_mask.setOn(
              NarrowbandLeaf::coordToOffset(adjacent));
        }
      }
    }

    leaf_work.second_candidate_offset = second_candidates.size();
    for (auto iterator = leaf_work.second_mask.beginOn(); iterator;
         ++iterator) {
      const Coord coordinate =
          leaf_work.origin + NarrowbandLeaf::offsetToLocalCoord(iterator.pos());
      if (index_accessor.isValueOn(coordinate))
        continue;
      second_candidates.push_back(
          {coordinate[0], coordinate[1], coordinate[2], 6,
           leaf_work.fragment_offset, leaf_work.fragment_count});
    }
    leaf_work.second_candidate_count =
        second_candidates.size() - leaf_work.second_candidate_offset;
  }

  vector<NarrowbandDistance> second_distances;
  evaluate_narrowband_distances(
      source_mesh, scale, points, triangles, voxel_size, fragments,
      second_candidates, second_distances);
  for (const NarrowbandLeafWork &leaf_work : work) {
    for (size_t local = 0; local < leaf_work.second_candidate_count;
         ++local) {
      const size_t candidate_index =
          leaf_work.second_candidate_offset + local;
      const NarrowbandCandidate &candidate =
          second_candidates[candidate_index];
      const Coord coordinate(candidate.x, candidate.y, candidate.z);
      if (!update_narrowband_voxel(
              coordinate, second_distances[candidate_index],
              distance_accessor, index_accessor, exterior_width,
              interior_width, voxel_size)) {
        continue;
      }
      for (int neighbour = 0; neighbour < 6; ++neighbour) {
        next_mask_accessor.setValueOn(
            coordinate + util::COORD_OFFSETS[neighbour]);
      }
    }
  }
  mask_tree.clear();
  mask_tree.merge(next_mask_tree);
}

void evaluate_narrowband_distances_cpu(
    const vector<Vec3s> &points, const vector<Vec3I> &triangles,
    double voxel_size, const vector<NarrowbandFragment> &fragments,
    const vector<NarrowbandCandidate> &candidates,
    vector<NarrowbandDistance> &distances) {
  distances.resize(candidates.size());
  for (size_t candidate_index = 0;
       candidate_index < candidates.size(); ++candidate_index) {
    const NarrowbandCandidate &candidate = candidates[candidate_index];
    const Vec3d center(candidate.x, candidate.y, candidate.z);
    double closest_distance = numeric_limits<double>::max();
    int closest_triangle_index = 0;
    int last_triangle_index = Int32(util::INVALID_IDX);
    const size_t end =
        candidate.fragment_offset + candidate.fragment_count;
    for (size_t fragment_index = candidate.fragment_offset;
         fragment_index < end; ++fragment_index) {
      const NarrowbandFragment &fragment = fragments[fragment_index];
      if (last_triangle_index == fragment.triangle_index)
        continue;
      const int manhattan = abs(fragment.x - candidate.x) +
                            abs(fragment.y - candidate.y) +
                            abs(fragment.z - candidate.z);
      if (manhattan > candidate.manhattan_limit)
        continue;
      last_triangle_index = fragment.triangle_index;
      const Vec3I &triangle = triangles[fragment.triangle_index];
      const Vec3d a(points[triangle[0]]);
      const Vec3d b(points[triangle[1]]);
      const Vec3d c(points[triangle[2]]);
      Vec3d uvw;
      const double distance =
          (center - math::closestPointOnTriangleToPoint(
                        a, c, b, center, uvw))
              .lengthSqr();
      if (distance < closest_distance) {
        closest_distance = distance;
        closest_triangle_index = fragment.triangle_index;
      }
    }
    distances[candidate_index] =
        {sqrt(closest_distance) * voxel_size, closest_triangle_index};
  }
}

struct NarrowbandBatchRequest {
  NarrowbandEvaluationInput input;
  bool done = false;
  exception_ptr error;
};

class NarrowbandCudaBatcher {
public:
  void evaluate(const NarrowbandEvaluationInput &input) {
    if (input.candidates->empty()) {
      input.distances->clear();
      return;
    }
    auto request = make_shared<NarrowbandBatchRequest>();
    request->input = input;
    bool leader = false;
    {
      lock_guard<mutex> lock(mutex_);
      queue_.push_back(request);
      if (!processing_) {
        processing_ = true;
        leader = true;
      }
      condition_.notify_all();
    }

    if (!leader) {
      unique_lock<mutex> lock(mutex_);
      condition_.wait(lock, [&]() { return request->done; });
      if (request->error)
        rethrow_exception(request->error);
      return;
    }

    while (true) {
      vector<shared_ptr<NarrowbandBatchRequest>> batch;
      {
        unique_lock<mutex> lock(mutex_);
        condition_.wait_for(lock, chrono::microseconds(250), [&]() {
          return queue_.size() >= 32;
        });
        const size_t count = min<size_t>(200, queue_.size());
        batch.reserve(count);
        for (size_t index = 0; index < count; ++index) {
          batch.push_back(move(queue_.front()));
          queue_.pop_front();
        }
      }

      exception_ptr error;
      try {
        vector<NarrowbandEvaluationInput> inputs;
        inputs.reserve(batch.size());
        size_t candidate_count = 0;
        for (const auto &pending : batch) {
          inputs.push_back(pending->input);
          candidate_count += pending->input.candidates->size();
        }
        if (environment_enabled("VISACD_PREPROCESS_EXPAND_TRACE")) {
          cerr << "[visacd expand] meshes=" << batch.size()
               << " candidates=" << candidate_count << '\n';
        }
        evaluate_narrowband_distances_cuda_batch(inputs);
      } catch (...) {
        error = current_exception();
      }

      bool finished = false;
      {
        lock_guard<mutex> lock(mutex_);
        for (const auto &pending : batch) {
          pending->error = error;
          pending->done = true;
        }
        if (queue_.empty()) {
          processing_ = false;
          finished = true;
        }
        condition_.notify_all();
      }
      if (finished)
        break;
    }

    if (request->error)
      rethrow_exception(request->error);
  }

private:
  mutex mutex_;
  condition_variable condition_;
  deque<shared_ptr<NarrowbandBatchRequest>> queue_;
  bool processing_ = false;
};

NarrowbandCudaBatcher &narrowband_cuda_batcher() {
  static NarrowbandCudaBatcher batcher;
  return batcher;
}

struct DenseNarrowbandBatchRequest {
  DenseNarrowbandInput input;
  bool done = false;
  exception_ptr error;
};

class DenseNarrowbandCudaBatcher {
public:
  void expand(const DenseNarrowbandInput &input) {
    auto request = make_shared<DenseNarrowbandBatchRequest>();
    request->input = input;
    bool leader = false;
    {
      lock_guard<mutex> lock(mutex_);
      queue_.push_back(request);
      if (!processing_) {
        processing_ = true;
        leader = true;
      }
      condition_.notify_all();
    }
    if (!leader) {
      unique_lock<mutex> lock(mutex_);
      condition_.wait(lock, [&]() { return request->done; });
      if (request->error)
        rethrow_exception(request->error);
      return;
    }

    while (true) {
      vector<shared_ptr<DenseNarrowbandBatchRequest>> batch;
      {
        unique_lock<mutex> lock(mutex_);
        condition_.wait_for(
            lock,
            chrono::microseconds(dense_narrowband_wait_microseconds()),
            [&]() { return queue_.size() >= 32; });
        const size_t count = min<size_t>(200, queue_.size());
        batch.reserve(count);
        for (size_t index = 0; index < count; ++index) {
          batch.push_back(move(queue_.front()));
          queue_.pop_front();
        }
      }
      exception_ptr error;
      try {
        vector<DenseNarrowbandInput> inputs;
        inputs.reserve(batch.size());
        size_t cell_count = 0;
        for (const auto &pending : batch) {
          inputs.push_back(pending->input);
          cell_count += pending->input.grid->active.size();
        }
        if (environment_enabled("VISACD_PREPROCESS_EXPAND_TRACE")) {
          cerr << "[visacd expand dense] meshes=" << batch.size()
               << " cells=" << cell_count << '\n';
        }
        expand_narrowband_dense_cuda_batch(inputs);
      } catch (...) {
        error = current_exception();
      }
      bool finished = false;
      {
        lock_guard<mutex> lock(mutex_);
        for (const auto &pending : batch) {
          pending->error = error;
          pending->done = true;
        }
        if (queue_.empty()) {
          processing_ = false;
          finished = true;
        }
        condition_.notify_all();
      }
      if (finished)
        break;
    }
    if (request->error)
      rethrow_exception(request->error);
  }

private:
  mutex mutex_;
  condition_variable condition_;
  deque<shared_ptr<DenseNarrowbandBatchRequest>> queue_;
  bool processing_ = false;
};

DenseNarrowbandCudaBatcher &dense_narrowband_cuda_batcher() {
  static DenseNarrowbandCudaBatcher batcher;
  return batcher;
}

struct SparseRenormalizeBatchRequest {
  SparseRenormalizeGrid *grid = nullptr;
  bool done = false;
  exception_ptr error;
};

class SparseRenormalizeCudaBatcher {
public:
  void renormalize(SparseRenormalizeGrid &grid) {
    auto request = make_shared<SparseRenormalizeBatchRequest>();
    request->grid = &grid;
    bool leader = false;
    {
      lock_guard<mutex> lock(mutex_);
      queue_.push_back(request);
      if (!processing_) {
        processing_ = true;
        leader = true;
      }
      condition_.notify_all();
    }
    if (!leader) {
      unique_lock<mutex> lock(mutex_);
      condition_.wait(lock, [&]() { return request->done; });
      if (request->error)
        rethrow_exception(request->error);
      return;
    }

    while (true) {
      vector<shared_ptr<SparseRenormalizeBatchRequest>> batch;
      {
        unique_lock<mutex> lock(mutex_);
        condition_.wait_for(lock, chrono::microseconds(250), [&]() {
          return queue_.size() >= 32;
        });
        const size_t count = min<size_t>(200, queue_.size());
        batch.reserve(count);
        for (size_t index = 0; index < count; ++index) {
          batch.push_back(move(queue_.front()));
          queue_.pop_front();
        }
      }
      exception_ptr error;
      try {
        vector<SparseRenormalizeGrid *> grids;
        grids.reserve(batch.size());
        size_t leaf_count = 0;
        for (const auto &pending : batch) {
          grids.push_back(pending->grid);
          leaf_count += pending->grid->active.size() /
                        NarrowbandLeaf::SIZE;
        }
        if (environment_enabled(
                "VISACD_PREPROCESS_RENORMALIZE_TRACE")) {
          cerr << "[visacd renormalize] meshes=" << batch.size()
               << " leaves=" << leaf_count << '\n';
        }
        renormalize_sparse_cuda_batch(grids);
      } catch (...) {
        error = current_exception();
      }
      bool finished = false;
      {
        lock_guard<mutex> lock(mutex_);
        for (const auto &pending : batch) {
          pending->error = error;
          pending->done = true;
        }
        if (queue_.empty()) {
          processing_ = false;
          finished = true;
        }
        condition_.notify_all();
      }
      if (finished)
        break;
    }
    if (request->error)
      rethrow_exception(request->error);
  }

private:
  mutex mutex_;
  condition_variable condition_;
  deque<shared_ptr<SparseRenormalizeBatchRequest>> queue_;
  bool processing_ = false;
};

SparseRenormalizeCudaBatcher &sparse_renormalize_cuda_batcher() {
  static SparseRenormalizeCudaBatcher batcher;
  return batcher;
}

bool renormalize_sparse_cuda(
    NarrowbandTree &distance_tree,
    const vector<NarrowbandLeaf *> &nodes, double voxel_size) {
  if (nodes.empty())
    return true;
  if (nodes.size() >
      static_cast<size_t>(numeric_limits<int>::max()))
    return false;
  SparseRenormalizeGrid grid;
  grid.voxel_size = voxel_size;
  const size_t cell_count =
      nodes.size() * static_cast<size_t>(NarrowbandLeaf::SIZE);
  const size_t neighbour_count = nodes.size() * 6;
  grid.active.resize(cell_count);
  grid.values.resize(cell_count);
  grid.neighbour_indices.resize(neighbour_count, -1);
  grid.neighbour_values.resize(neighbour_count);

  unordered_map<const NarrowbandLeaf *, int> leaf_indices;
  leaf_indices.reserve(nodes.size());
  for (size_t leaf = 0; leaf < nodes.size(); ++leaf)
    leaf_indices.emplace(nodes[leaf], static_cast<int>(leaf));

  tree::ValueAccessor<NarrowbandTree> accessor(distance_tree);
  const int axis[6] = {0, 0, 1, 1, 2, 2};
  const int direction[6] = {-1, 1, -1, 1, -1, 1};
  for (size_t leaf = 0; leaf < nodes.size(); ++leaf) {
    const NarrowbandLeaf &node = *nodes[leaf];
    const size_t cell_offset =
        leaf * static_cast<size_t>(NarrowbandLeaf::SIZE);
    copy_n(node.buffer().data(), NarrowbandLeaf::SIZE,
           grid.values.begin() + cell_offset);
    const auto &mask = node.getValueMask();
    for (Index position = 0; position < NarrowbandLeaf::SIZE;
         ++position) {
      grid.active[cell_offset + position] =
          mask.isOn(position) ? 1 : 0;
    }
    for (int neighbour = 0; neighbour < 6; ++neighbour) {
      Coord origin = node.origin();
      origin[axis[neighbour]] +=
          direction[neighbour] * NarrowbandLeaf::DIM;
      const NarrowbandLeaf *adjacent =
          accessor.probeConstLeaf(origin);
      const auto found = leaf_indices.find(adjacent);
      const size_t packed = leaf * 6 + neighbour;
      if (adjacent) {
        if (found == leaf_indices.end())
          throw logic_error(
              "Sparse renormalization leaf topology diverged");
        grid.neighbour_indices[packed] = found->second;
      } else {
        grid.neighbour_values[packed] =
            accessor.getValue(origin);
      }
    }
  }

  sparse_renormalize_cuda_batcher().renormalize(grid);
  for (size_t leaf = 0; leaf < nodes.size(); ++leaf) {
    NarrowbandLeaf &node = *nodes[leaf];
    double *values = node.buffer().data();
    const size_t cell_offset =
        leaf * static_cast<size_t>(NarrowbandLeaf::SIZE);
    for (auto iterator = node.cbeginValueOn(); iterator; ++iterator)
      values[iterator.pos()] =
          grid.values[cell_offset + iterator.pos()];
  }
  return true;
}

struct SparseSurfacePostBatchRequest {
  SparseSurfacePostGrid *grid = nullptr;
  bool done = false;
  exception_ptr error;
};

class SparseSurfacePostCudaBatcher {
public:
  void process(SparseSurfacePostGrid &grid) {
    auto request = make_shared<SparseSurfacePostBatchRequest>();
    request->grid = &grid;
    bool leader = false;
    {
      lock_guard<mutex> lock(mutex_);
      queue_.push_back(request);
      if (!processing_) {
        processing_ = true;
        leader = true;
      }
      condition_.notify_all();
    }
    if (!leader) {
      unique_lock<mutex> lock(mutex_);
      condition_.wait(lock, [&]() { return request->done; });
      if (request->error)
        rethrow_exception(request->error);
      return;
    }
    while (true) {
      vector<shared_ptr<SparseSurfacePostBatchRequest>> batch;
      {
        unique_lock<mutex> lock(mutex_);
        condition_.wait_for(lock, chrono::microseconds(250), [&]() {
          return queue_.size() >= 32;
        });
        const size_t count = min<size_t>(200, queue_.size());
        batch.reserve(count);
        for (size_t index = 0; index < count; ++index) {
          batch.push_back(move(queue_.front()));
          queue_.pop_front();
        }
      }
      exception_ptr error;
      try {
        vector<SparseSurfacePostGrid *> grids;
        grids.reserve(batch.size());
        size_t leaf_count = 0;
        for (const auto &pending : batch) {
          grids.push_back(pending->grid);
          leaf_count += pending->grid->active.size() /
                        NarrowbandLeaf::SIZE;
        }
        if (environment_enabled("VISACD_PREPROCESS_SIGN_TRACE")) {
          cerr << "[visacd surface post] meshes=" << batch.size()
               << " leaves=" << leaf_count << '\n';
        }
        postprocess_sparse_surface_cuda_batch(grids);
      } catch (...) {
        error = current_exception();
      }
      bool finished = false;
      {
        lock_guard<mutex> lock(mutex_);
        for (const auto &pending : batch) {
          pending->error = error;
          pending->done = true;
        }
        if (queue_.empty()) {
          processing_ = false;
          finished = true;
        }
        condition_.notify_all();
      }
      if (finished)
        break;
    }
    if (request->error)
      rethrow_exception(request->error);
  }

private:
  mutex mutex_;
  condition_variable condition_;
  deque<shared_ptr<SparseSurfacePostBatchRequest>> queue_;
  bool processing_ = false;
};

SparseSurfacePostCudaBatcher &sparse_surface_post_cuda_batcher() {
  static SparseSurfacePostCudaBatcher batcher;
  return batcher;
}

bool postprocess_sparse_surface_cuda(
    NarrowbandTree &distance_tree, NarrowbandIndexTree &index_tree,
    const vector<NarrowbandLeaf *> &nodes, const Mesh &source_mesh,
    double scale) {
  if (nodes.empty())
    return true;
  if (nodes.size() >
      static_cast<size_t>(numeric_limits<int>::max()))
    return false;
  SparseSurfacePostGrid grid;
  grid.mesh = &source_mesh;
  grid.scale = scale;
  const size_t cell_count =
      nodes.size() * static_cast<size_t>(NarrowbandLeaf::SIZE);
  grid.leaf_origins.resize(nodes.size() * 3);
  grid.active.resize(cell_count);
  grid.values.resize(cell_count);
  grid.triangle_indices.resize(
      cell_count, Int32(util::INVALID_IDX));
  grid.neighbour_indices.resize(nodes.size() * 27, -1);
  grid.neighbour_values.resize(nodes.size() * 27);

  unordered_map<const NarrowbandLeaf *, int> leaf_indices;
  leaf_indices.reserve(nodes.size());
  for (size_t leaf = 0; leaf < nodes.size(); ++leaf)
    leaf_indices.emplace(nodes[leaf], static_cast<int>(leaf));
  tree::ValueAccessor<NarrowbandTree> distance_accessor(distance_tree);
  tree::ValueAccessor<NarrowbandIndexTree> index_accessor(index_tree);
  for (size_t leaf = 0; leaf < nodes.size(); ++leaf) {
    const NarrowbandLeaf &distance_leaf = *nodes[leaf];
    const auto *index_leaf =
        index_accessor.probeConstLeaf(distance_leaf.origin());
    if (!index_leaf)
      throw logic_error(
          "Sparse surface distance and index trees diverged");
    const Coord origin = distance_leaf.origin();
    for (int axis = 0; axis < 3; ++axis)
      grid.leaf_origins[leaf * 3 + axis] = origin[axis];
    const size_t cell_offset =
        leaf * static_cast<size_t>(NarrowbandLeaf::SIZE);
    copy_n(distance_leaf.buffer().data(), NarrowbandLeaf::SIZE,
           grid.values.begin() + cell_offset);
    copy_n(index_leaf->buffer().data(), NarrowbandLeaf::SIZE,
           grid.triangle_indices.begin() + cell_offset);
    const auto &mask = distance_leaf.getValueMask();
    for (Index position = 0; position < NarrowbandLeaf::SIZE;
         ++position) {
      grid.active[cell_offset + position] =
          mask.isOn(position) ? 1 : 0;
    }
    size_t slot = 0;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz, ++slot) {
          Coord adjacent_origin = origin;
          adjacent_origin[0] += dx * NarrowbandLeaf::DIM;
          adjacent_origin[1] += dy * NarrowbandLeaf::DIM;
          adjacent_origin[2] += dz * NarrowbandLeaf::DIM;
          const NarrowbandLeaf *adjacent =
              distance_accessor.probeConstLeaf(adjacent_origin);
          const size_t packed = leaf * 27 + slot;
          if (adjacent) {
            const auto found = leaf_indices.find(adjacent);
            if (found == leaf_indices.end())
              throw logic_error(
                  "Sparse surface leaf topology diverged");
            grid.neighbour_indices[packed] = found->second;
          } else {
            grid.neighbour_values[packed] =
                distance_accessor.getValue(adjacent_origin);
          }
        }
      }
    }
  }

  sparse_surface_post_cuda_batcher().process(grid);
  for (size_t leaf = 0; leaf < nodes.size(); ++leaf) {
    NarrowbandLeaf &distance_leaf = *nodes[leaf];
    auto *index_leaf =
        index_accessor.probeLeaf(distance_leaf.origin());
    if (!index_leaf)
      throw logic_error(
          "Sparse surface distance and index trees diverged");
    double *values = distance_leaf.buffer().data();
    const size_t cell_offset =
        leaf * static_cast<size_t>(NarrowbandLeaf::SIZE);
    for (Index position = 0; position < NarrowbandLeaf::SIZE;
         ++position) {
      if (!distance_leaf.getValueMask().isOn(position))
        continue;
      values[position] = grid.values[cell_offset + position];
      if (!grid.active[cell_offset + position]) {
        distance_leaf.setValueOff(position);
        index_leaf->setValueOff(position);
      }
    }
  }
  return true;
}

bool expand_narrowband_dense(
    NarrowbandTree &distance_tree, NarrowbandIndexTree &index_tree,
    const Mesh &source_mesh, double scale, double exterior_width,
    double interior_width, double voxel_size,
    unsigned int maximum_iterations) {
  CoordBBox active_bounds;
  if (!distance_tree.evalActiveVoxelBoundingBox(active_bounds))
    return true;
  if (maximum_iterations >
      static_cast<unsigned int>((numeric_limits<int>::max() - 3) / 2))
    return false;
  const int padding =
      static_cast<int>(maximum_iterations) * 2 + 3;
  Coord minimum = active_bounds.min().offsetBy(-padding);
  Coord maximum = active_bounds.max().offsetBy(padding);
  minimum &= ~7;
  maximum |= 7;
  const Coord dimensions = maximum - minimum + Coord(1);
  size_t cell_count = static_cast<size_t>(dimensions[0]);
  if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] <= 0 ||
      cell_count > numeric_limits<size_t>::max() /
                       static_cast<size_t>(dimensions[1])) {
    return false;
  }
  cell_count *= static_cast<size_t>(dimensions[1]);
  if (cell_count > numeric_limits<size_t>::max() /
                       static_cast<size_t>(dimensions[2])) {
    return false;
  }
  cell_count *= static_cast<size_t>(dimensions[2]);

  DenseNarrowbandGrid grid;
  for (int axis = 0; axis < 3; ++axis) {
    grid.minimum[axis] = minimum[axis];
    grid.dimensions[axis] = dimensions[axis];
  }
  grid.exterior_width = exterior_width;
  grid.interior_width = interior_width;
  grid.voxel_size = voxel_size;
  grid.iterations = maximum_iterations;
  grid.active.resize(cell_count);
  grid.inside.resize(cell_count);
  grid.distances.resize(cell_count);
  grid.triangle_indices.resize(cell_count, Int32(util::INVALID_IDX));
  vector<unsigned char> original_active(cell_count);
  tree::ValueAccessor<NarrowbandTree> distance_accessor(distance_tree);
  tree::ValueAccessor<NarrowbandIndexTree> index_accessor(index_tree);
  const auto dense_offset = [&](int x, int y, int z) {
    return (static_cast<size_t>(x - minimum[0]) *
                static_cast<size_t>(dimensions[1]) +
            static_cast<size_t>(y - minimum[1])) *
               static_cast<size_t>(dimensions[2]) +
           static_cast<size_t>(z - minimum[2]);
  };
  Coord origin;
  for (int x = minimum[0]; x <= maximum[0];
       x += NarrowbandLeaf::DIM) {
    origin[0] = x;
    for (int y = minimum[1]; y <= maximum[1];
         y += NarrowbandLeaf::DIM) {
      origin[1] = y;
      for (int z = minimum[2]; z <= maximum[2];
           z += NarrowbandLeaf::DIM) {
        origin[2] = z;
        const NarrowbandLeaf *distance_leaf =
            distance_accessor.probeConstLeaf(origin);
        const NarrowbandIndexTree::LeafNodeType *index_leaf =
            index_accessor.probeConstLeaf(origin);
        if (!distance_leaf) {
          const double distance = distance_accessor.getValue(origin);
          const unsigned char inside = distance < 0.0 ? 1 : 0;
          for (int local_x = 0; local_x < NarrowbandLeaf::DIM;
               ++local_x) {
            for (int local_y = 0; local_y < NarrowbandLeaf::DIM;
                 ++local_y) {
              for (int local_z = 0; local_z < NarrowbandLeaf::DIM;
                   ++local_z) {
                const size_t offset =
                    dense_offset(x + local_x, y + local_y,
                                 z + local_z);
                grid.inside[offset] = inside;
                grid.distances[offset] = distance;
              }
            }
          }
          continue;
        }
        const auto &mask = distance_leaf->getValueMask();
        const double *distances = distance_leaf->buffer().data();
        const Int32 *indices =
            index_leaf ? index_leaf->buffer().data() : nullptr;
        for (int local_x = 0; local_x < NarrowbandLeaf::DIM;
             ++local_x) {
          for (int local_y = 0; local_y < NarrowbandLeaf::DIM;
               ++local_y) {
            for (int local_z = 0; local_z < NarrowbandLeaf::DIM;
                 ++local_z) {
              const Coord coordinate(x + local_x, y + local_y,
                                     z + local_z);
              const Index position =
                  NarrowbandLeaf::coordToOffset(coordinate);
              const size_t offset =
                  dense_offset(coordinate[0], coordinate[1],
                               coordinate[2]);
              const bool active = mask.isOn(position);
              if (active && !indices)
                throw logic_error(
                    "Narrowband distance and index trees diverged");
              grid.active[offset] = active ? 1 : 0;
              original_active[offset] = grid.active[offset];
              grid.inside[offset] =
                  distances[position] < 0.0 ? 1 : 0;
              grid.distances[offset] = distances[position];
              if (active)
                grid.triangle_indices[offset] = indices[position];
            }
          }
        }
      }
    }
  }

  dense_narrowband_cuda_batcher().expand({&source_mesh, scale, &grid});
  size_t offset = 0;
  Coord coordinate;
  for (int x = minimum[0]; x <= maximum[0]; ++x) {
    coordinate[0] = x;
    for (int y = minimum[1]; y <= maximum[1]; ++y) {
      coordinate[1] = y;
      for (int z = minimum[2]; z <= maximum[2]; ++z, ++offset) {
        if (!grid.active[offset] || original_active[offset])
          continue;
        coordinate[2] = z;
        distance_accessor.setValue(coordinate, grid.distances[offset]);
        index_accessor.setValue(coordinate,
                                grid.triangle_indices[offset]);
      }
    }
  }
  return true;
}

void evaluate_narrowband_distances(
    const Mesh &source_mesh, double scale, const vector<Vec3s> &points,
    const vector<Vec3I> &triangles, double voxel_size,
    const vector<NarrowbandFragment> &fragments,
    const vector<NarrowbandCandidate> &candidates,
    vector<NarrowbandDistance> &distances) {
  try {
    narrowband_cuda_batcher().evaluate(
        {&source_mesh, scale, voxel_size, &fragments, &candidates,
         &distances});
  } catch (...) {
    evaluate_narrowband_distances_cpu(
        points, triangles, voxel_size, fragments, candidates, distances);
    return;
  }
  if (!environment_enabled("VISACD_VERIFY_CUDA_PREPROCESS_EXPAND"))
    return;
  vector<NarrowbandDistance> reference;
  evaluate_narrowband_distances_cpu(
      points, triangles, voxel_size, fragments, candidates, reference);
  bool exact = reference.size() == distances.size();
  for (size_t index = 0; exact && index < reference.size(); ++index) {
    exact = reference[index].triangle_index ==
                distances[index].triangle_index &&
            memcmp(&reference[index].distance, &distances[index].distance,
                   sizeof(double)) == 0;
  }
  if (!exact)
    distances = move(reference);
}

// This is the post-voxelization portion of OpenVDB 8.2's MeshToVolume
// pipeline, specialized for the exact surface records produced on CUDA.
DoubleGrid::Ptr signed_distance_field_from_surface(
    const vector<Vec3s> &points, const vector<Vec3I> &triangles,
    const math::Transform &transform,
    const vector<SurfaceVoxelRecord> &surface, float exterior_band_width,
    float interior_band_width, ManifoldPreprocessMetrics *metrics,
    const Mesh &source_mesh, double scale) {
  using GridType = DoubleGrid;
  using TreeType = GridType::TreeType;
  using LeafNodeType = TreeType::LeafNodeType;
  using ValueType = GridType::ValueType;
  using IntGridType = GridType::ValueConverter<Int32>::Type;
  using IntTreeType = IntGridType::TreeType;
  using BoolTreeType = TreeType::ValueConverter<bool>::Type;
  using Adapter = tools::QuadAndTriangleDataAdapter<Vec3s, Vec3I>;

  const auto seed_grid_start = PreprocessClock::now();
  GridType::Ptr distance_grid(
      new GridType(numeric_limits<ValueType>::max()));
  distance_grid->setTransform(transform.copy());
  distance_grid->setGridClass(GRID_LEVEL_SET);

  IntGridType index_grid(Int32(util::INVALID_IDX));
  index_grid.setTransform(transform.copy());
  TreeType &distance_tree = distance_grid->tree();
  IntTreeType &index_tree = index_grid.tree();
  tree::ValueAccessor<TreeType> distance_accessor(distance_tree);
  tree::ValueAccessor<IntTreeType> index_accessor(index_tree);
  for (const SurfaceVoxelRecord &voxel : surface) {
    const Coord coordinate(voxel.x, voxel.y, voxel.z);
    distance_accessor.setValue(coordinate, voxel.squared_distance);
    index_accessor.setValue(coordinate, voxel.triangle_index);
  }

  const ValueType voxel_size = ValueType(transform.voxelSize()[0]);
  ValueType exterior_width = ValueType(exterior_band_width) * voxel_size;
  ValueType interior_width = ValueType(interior_band_width);
  if (interior_width < numeric_limits<ValueType>::max())
    interior_width *= voxel_size;
  Adapter mesh(points, triangles);
  if (metrics)
    metrics->sdf_seed_grid_ns = elapsed_ns(seed_grid_start);

  const auto trace_start = PreprocessClock::now();
  tools::traceExteriorBoundaries(distance_tree);
  if (metrics)
    metrics->sdf_trace_ns = elapsed_ns(trace_start);
  vector<LeafNodeType *> nodes;
  nodes.reserve(distance_tree.leafCount());
  distance_tree.getNodes(nodes);
  const tbb::blocked_range<size_t> node_range(0, nodes.size());
  const auto sign_start = PreprocessClock::now();
  auto cleanup_start = sign_start;
  bool cuda_surface_postprocessed = false;
  if (environment_enabled("VISACD_ENABLE_CUDA_PREPROCESS_SIGN")) {
    try {
      cuda_surface_postprocessed = postprocess_sparse_surface_cuda(
          distance_tree, index_tree, nodes, source_mesh, scale);
    } catch (const exception &error) {
      if (environment_enabled("VISACD_PREPROCESS_SIGN_TRACE"))
        cerr << "[visacd surface post] fallback=" << error.what()
             << '\n';
      cuda_surface_postprocessed = false;
    } catch (...) {
      if (environment_enabled("VISACD_PREPROCESS_SIGN_TRACE"))
        cerr << "[visacd surface post] fallback=unknown exception\n";
      cuda_surface_postprocessed = false;
    }
  }
  if (cuda_surface_postprocessed) {
    if (metrics) {
      metrics->sdf_sign_ns = elapsed_ns(sign_start);
      metrics->sdf_validate_ns = 0;
    }
    cleanup_start = PreprocessClock::now();
  } else {
    using SignOperation =
        tools::mesh_to_volume_internal::ComputeIntersectingVoxelSign<
            TreeType, Adapter>;
    tbb::parallel_for(
        node_range,
        SignOperation(nodes, distance_tree, index_tree, mesh));
    if (metrics)
      metrics->sdf_sign_ns = elapsed_ns(sign_start);
    const auto validate_start = PreprocessClock::now();
    tbb::parallel_for(
        node_range,
        tools::mesh_to_volume_internal::
            ValidateIntersectingVoxels<TreeType>(
                distance_tree, nodes));
    if (metrics)
      metrics->sdf_validate_ns = elapsed_ns(validate_start);
    cleanup_start = PreprocessClock::now();
    tbb::parallel_for(
        node_range,
        tools::mesh_to_volume_internal::
            RemoveSelfIntersectingSurface<TreeType>(
                nodes, distance_tree, index_tree));
  }
  tools::pruneInactive(distance_tree, true);
  tools::pruneInactive(index_tree, true);
  if (metrics)
    metrics->sdf_cleanup_ns = elapsed_ns(cleanup_start);

  if (distance_tree.activeVoxelCount() == 0) {
    distance_tree.clear();
    distance_tree.root().setBackground(exterior_width, false);
    return distance_grid;
  }

  nodes.clear();
  nodes.reserve(distance_tree.leafCount());
  distance_tree.getNodes(nodes);
  const auto transform_flood_start = PreprocessClock::now();
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, nodes.size()),
      tools::mesh_to_volume_internal::TransformValues<TreeType>(nodes,
                                                                 voxel_size,
                                                                 false));
  distance_tree.root().setBackground(exterior_width, false);
  tools::signedFloodFillWithValues(distance_tree, exterior_width,
                                   -interior_width);
  if (metrics)
    metrics->sdf_transform_flood_ns =
        elapsed_ns(transform_flood_start);

  const ValueType minimum_band_width = voxel_size * ValueType(2.0);
  const auto expand_start = PreprocessClock::now();
  if (interior_width > minimum_band_width ||
      exterior_width > minimum_band_width) {
    BoolTreeType mask_tree(false);
    nodes.clear();
    nodes.reserve(distance_tree.leafCount());
    distance_tree.getNodes(nodes);
    tools::mesh_to_volume_internal::ConstructVoxelMask<TreeType> mask_builder(
        mask_tree, distance_tree, nodes);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()),
                         mask_builder);

    unsigned maximum_iterations = numeric_limits<unsigned>::max();
    const double estimated =
        2.0 * ceil((max(interior_width, exterior_width) -
                    minimum_band_width) /
                   voxel_size);
    if (estimated < double(maximum_iterations))
      maximum_iterations = unsigned(estimated);

    bool dense_expanded = false;
    if (environment_enabled(
            "VISACD_ENABLE_CUDA_PREPROCESS_EXPAND_DENSE")) {
      try {
        dense_expanded = expand_narrowband_dense(
            distance_tree, index_tree, source_mesh, scale,
            exterior_width, interior_width, voxel_size,
            maximum_iterations);
      } catch (const exception &error) {
        if (environment_enabled("VISACD_PREPROCESS_EXPAND_TRACE"))
          cerr << "[visacd expand dense] fallback=" << error.what()
               << '\n';
        dense_expanded = false;
      } catch (...) {
        if (environment_enabled("VISACD_PREPROCESS_EXPAND_TRACE"))
          cerr << "[visacd expand dense] fallback=unknown exception\n";
        dense_expanded = false;
      }
    }

    if (!dense_expanded) {
      vector<BoolTreeType::LeafNodeType *> mask_nodes;
      unsigned iteration = 0;
      while (true) {
        const size_t mask_node_count = mask_tree.leafCount();
        if (mask_node_count == 0)
          break;
        mask_nodes.clear();
        mask_nodes.reserve(mask_node_count);
        mask_tree.getNodes(mask_nodes);
        const tbb::blocked_range<size_t> range(0, mask_nodes.size());
        tbb::parallel_for(
            range,
            tools::mesh_to_volume_internal::DiffLeafNodeMask<TreeType>(
                distance_tree, mask_nodes));
        if (environment_enabled("VISACD_ENABLE_CUDA_PREPROCESS_EXPAND")) {
          expand_narrowband_cuda_iteration(
              distance_tree, index_tree, mask_tree, mask_nodes,
              source_mesh, scale, points, triangles, exterior_width,
              interior_width, voxel_size);
        } else {
          tools::mesh_to_volume_internal::expandNarrowband(
              distance_tree, index_tree, mask_tree, mask_nodes, mesh,
              exterior_width, interior_width, voxel_size);
        }
        if (++iteration >= maximum_iterations)
          break;
      }
    }
  }
  if (metrics)
    metrics->sdf_expand_ns = elapsed_ns(expand_start);

  const auto renormalize_start = PreprocessClock::now();
  index_grid.clear();
  nodes.clear();
  nodes.reserve(distance_tree.leafCount());
  distance_tree.getNodes(nodes);
  const ValueType offset = ValueType(0.8 * voxel_size);
  const tbb::blocked_range<size_t> final_node_range(0, nodes.size());
  bool cuda_renormalized = false;
  if (environment_enabled(
          "VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE")) {
    try {
      cuda_renormalized = renormalize_sparse_cuda(
          distance_tree, nodes, voxel_size);
    } catch (const exception &error) {
      if (environment_enabled(
              "VISACD_PREPROCESS_RENORMALIZE_TRACE")) {
        cerr << "[visacd renormalize] fallback=" << error.what()
             << '\n';
      }
      cuda_renormalized = false;
    } catch (...) {
      if (environment_enabled(
              "VISACD_PREPROCESS_RENORMALIZE_TRACE")) {
        cerr << "[visacd renormalize] fallback=unknown exception\n";
      }
      cuda_renormalized = false;
    }
  }
  if (!cuda_renormalized) {
    unique_ptr<ValueType[]> buffer(
        new ValueType[LeafNodeType::SIZE * nodes.size()]);
    tbb::parallel_for(
        final_node_range,
        tools::mesh_to_volume_internal::OffsetValues<TreeType>(
            nodes, -offset));
    tbb::parallel_for(
        final_node_range,
        tools::mesh_to_volume_internal::Renormalize<TreeType>(
            distance_tree, nodes, buffer.get(), voxel_size));
    tbb::parallel_for(
        final_node_range,
        tools::mesh_to_volume_internal::MinCombine<TreeType>(
            nodes, buffer.get()));
  }
  tbb::parallel_for(
      final_node_range,
      tools::mesh_to_volume_internal::OffsetValues<TreeType>(
          nodes,
          offset -
              tools::mesh_to_volume_internal::Tolerance<ValueType>::epsilon()));

  if (min(interior_width, exterior_width) <
      voxel_size * ValueType(4.0)) {
    tbb::parallel_for(
        final_node_range,
        tools::mesh_to_volume_internal::InactivateValues<TreeType>(
            nodes, exterior_width, interior_width));
    tools::pruneLevelSet(distance_tree, exterior_width, -interior_width);
  }
  if (metrics)
    metrics->sdf_renormalize_ns = elapsed_ns(renormalize_start);
  return distance_grid;
}

bool meshes_match_exactly(const Mesh &first, const Mesh &second) {
  if (first.vertices.size() != second.vertices.size() ||
      first.triangles.size() != second.triangles.size())
    return false;
  for (size_t index = 0; index < first.vertices.size(); ++index) {
    for (int axis = 0; axis < 3; ++axis) {
      if (memcmp(&first.vertices[index][axis],
                 &second.vertices[index][axis], sizeof(double)) != 0)
        return false;
    }
  }
  return first.triangles == second.triangles;
}

} // namespace

vector<SurfaceVoxelRecord>
reference_surface_voxelization(const Mesh &input, double scale) {
  vector<Vec3s> points;
  vector<Vec3I> triangles;
  points.reserve(input.vertices.size());
  triangles.reserve(input.triangles.size());
  for (const Vec3D &vertex : input.vertices) {
    points.push_back({static_cast<float>(vertex[0] * scale),
                      static_cast<float>(vertex[1] * scale),
                      static_cast<float>(vertex[2] * scale)});
  }
  for (const array<int, 3> &triangle : input.triangles) {
    triangles.push_back({static_cast<unsigned int>(triangle[0]),
                         static_cast<unsigned int>(triangle[1]),
                         static_cast<unsigned int>(triangle[2])});
  }
  if (points.empty() || triangles.empty())
    return {};

  using TreeType = DoubleGrid::TreeType;
  using IntTreeType = TreeType::ValueConverter<Int32>::Type;
  using VoxelizationData =
      tools::mesh_to_volume_internal::VoxelizationData<TreeType>;
  using DataTable =
      tbb::enumerable_thread_specific<typename VoxelizationData::Ptr>;
  using Adapter = tools::QuadAndTriangleDataAdapter<Vec3s, Vec3I>;
  using Voxelizer =
      tools::mesh_to_volume_internal::VoxelizePolygons<TreeType, Adapter>;

  Adapter adapter(points, triangles);
  DataTable data;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, adapter.polygonCount()),
                    Voxelizer(data, adapter));

  TreeType distance_tree(numeric_limits<double>::max());
  IntTreeType index_tree(Int32(util::INVALID_IDX));
  for (auto iterator = data.begin(); iterator != data.end(); ++iterator) {
    VoxelizationData &item = **iterator;
    tools::mesh_to_volume_internal::combineData(
        distance_tree, index_tree, item.distTree, item.indexTree);
  }

  tree::ValueAccessor<const IntTreeType> index_accessor(index_tree);
  vector<SurfaceVoxelRecord> records;
  records.reserve(distance_tree.activeVoxelCount());
  for (auto iterator = distance_tree.cbeginValueOn(); iterator; ++iterator) {
    const Coord coordinate = iterator.getCoord();
    records.push_back({coordinate.x(), coordinate.y(), coordinate.z(),
                       *iterator, index_accessor.getValue(coordinate)});
  }
  sort(records.begin(), records.end(), [](const auto &first,
                                          const auto &second) {
    if (first.x != second.x)
      return first.x < second.x;
    if (first.y != second.y)
      return first.y < second.y;
    return first.z < second.z;
  });
  return records;
}

bool sdf_manifold(Mesh &input, Mesh &output, double scale, double level_set,
                  bool use_cuda, string *fallback_reason,
                  ManifoldPreprocessMetrics *metrics,
                  const vector<SurfaceVoxelRecord> *provided_surface =
                      nullptr) {
  const auto marshal_start = PreprocessClock::now();
  vector<Vec3s> points;
  vector<Vec3I> tris;
  vector<Vec4I> quads;

  points.reserve(input.vertices.size());
  tris.reserve(input.triangles.size());

  for (unsigned int i = 0; i < input.vertices.size(); ++i) {
    points.push_back({(float)(input.vertices[i][0] * scale),
                      (float)(input.vertices[i][1] * scale),
                      (float)(input.vertices[i][2] * scale)});
  }
  for (unsigned int i = 0; i < input.triangles.size(); ++i) {
    tris.push_back({(unsigned int)input.triangles[i][0],
                    (unsigned int)input.triangles[i][1],
                    (unsigned int)input.triangles[i][2]});
  }

  math::Transform::Ptr xform = math::Transform::createLinearTransform();
  tools::QuadAndTriangleDataAdapter<Vec3s, Vec3I> mesh(points, tris);
  if (metrics) {
    metrics->input_vertices = input.vertices.size();
    metrics->input_triangles = input.triangles.size();
    metrics->marshal_input_ns = elapsed_ns(marshal_start);
  }

  const auto sdf_start = PreprocessClock::now();
  DoubleGrid::Ptr sgrid;
  if (provided_surface) {
    sgrid = signed_distance_field_from_surface(
        points, tris, *xform, *provided_surface,
        static_cast<float>(level_set * scale + 1.0), 3.0f, metrics,
        input, scale);
  } else if (use_cuda) {
    try {
      static thread_local ManifoldCudaRuntime runtime;
      SurfaceVoxelizationResult surface =
          runtime.voxelize_surface(input, scale);
      if (!surface.supported) {
        if (fallback_reason)
          *fallback_reason = surface.fallback_reason;
        return false;
      }
      sgrid = signed_distance_field_from_surface(
          points, tris, *xform, surface.records,
          static_cast<float>(level_set * scale + 1.0), 3.0f, metrics,
          input, scale);
    } catch (const exception &error) {
      if (fallback_reason)
        *fallback_reason = error.what();
      return false;
    }
  } else {
    sgrid = tools::meshToSignedDistanceField<DoubleGrid>(
        *xform, points, tris, quads, level_set * scale + 1.0, 3.0);
  }
  if (metrics) {
    metrics->mesh_to_sdf_ns = elapsed_ns(sdf_start);
    metrics->active_voxels = sgrid->activeVoxelCount();
  }

  vector<Vec3s> newPoints;
  vector<Vec3I> newTriangles;
  vector<Vec4I> newQuads;
  const auto meshing_start = PreprocessClock::now();
  tools::volumeToMesh(*sgrid, newPoints, newTriangles, newQuads, level_set*scale);
  if (metrics)
    metrics->volume_to_mesh_ns = elapsed_ns(meshing_start);

  const auto output_start = PreprocessClock::now();
  output.clear();
  output.vertices.reserve(newPoints.size());
  output.triangles.reserve(newTriangles.size() + newQuads.size() * 2);
  for (unsigned int i = 0; i < newPoints.size(); ++i) {
    output.vertices.push_back({newPoints[i][0] / scale, newPoints[i][1] / scale,
                               newPoints[i][2] / scale});
  }
  for (unsigned int i = 0; i < newTriangles.size(); ++i) {
    output.triangles.push_back({(int)newTriangles[i][0],
                                (int)newTriangles[i][2],
                                (int)newTriangles[i][1]});
  }
  for (unsigned int i = 0; i < newQuads.size(); ++i) {
    output.triangles.push_back(
        {(int)newQuads[i][0], (int)newQuads[i][2], (int)newQuads[i][1]});
    output.triangles.push_back(
        {(int)newQuads[i][0], (int)newQuads[i][3], (int)newQuads[i][2]});
  }
  if (metrics) {
    metrics->marshal_output_ns = elapsed_ns(output_start);
    metrics->output_vertices = output.vertices.size();
    metrics->output_triangles = output.triangles.size();
  }
  return true;
}

void manifold_preprocess_cpu_reference(Mesh &m, double scale,
                                       double level_set,
                                       ManifoldPreprocessMetrics *metrics) {
  if (metrics)
    *metrics = ManifoldPreprocessMetrics{};
  const auto copy_start = PreprocessClock::now();
  Mesh tmp = m;
  if (metrics)
    metrics->copy_input_ns = elapsed_ns(copy_start);
  Mesh output;
  sdf_manifold(tmp, output, scale, level_set, false, nullptr, metrics);
  m = std::move(output);
}

bool manifold_preprocess_cuda_candidate(
    Mesh &m, double scale, double level_set, string *fallback_reason,
    ManifoldPreprocessMetrics *metrics) {
  if (metrics)
    *metrics = ManifoldPreprocessMetrics{};
  const auto copy_start = PreprocessClock::now();
  Mesh tmp = m;
  if (metrics)
    metrics->copy_input_ns = elapsed_ns(copy_start);
  Mesh output;
  if (!sdf_manifold(tmp, output, scale, level_set, true,
                    fallback_reason, metrics))
    return false;
  m = std::move(output);
  return true;
}

void manifold_preprocess_from_surface_records(
    Mesh &m, const vector<SurfaceVoxelRecord> &surface, double scale,
    double level_set, ManifoldPreprocessMetrics *metrics) {
  if (metrics)
    *metrics = ManifoldPreprocessMetrics{};
  const auto copy_start = PreprocessClock::now();
  Mesh tmp = m;
  if (metrics)
    metrics->copy_input_ns = elapsed_ns(copy_start);
  Mesh output;
  sdf_manifold(tmp, output, scale, level_set, false, nullptr, metrics,
               &surface);
  m = std::move(output);
}

void manifold_preprocess(Mesh &m, double scale, double level_set,
                         ManifoldPreprocessMetrics *metrics) {
  if (!environment_enabled("VISACD_ENABLE_CUDA_PREPROCESS")) {
    manifold_preprocess_cpu_reference(m, scale, level_set, metrics);
    return;
  }

  if (environment_enabled("VISACD_VERIFY_CUDA_PREPROCESS")) {
    Mesh reference = m;
    Mesh candidate = m;
    ManifoldPreprocessMetrics candidate_metrics;
    string fallback_reason;
    if (manifold_preprocess_cuda_candidate(
            candidate, scale, level_set, &fallback_reason,
            metrics ? &candidate_metrics : nullptr)) {
      ManifoldPreprocessMetrics reference_metrics;
      manifold_preprocess_cpu_reference(
          reference, scale, level_set,
          metrics ? &reference_metrics : nullptr);
      if (meshes_match_exactly(reference, candidate)) {
        m = std::move(candidate);
        if (metrics)
          *metrics = candidate_metrics;
        return;
      }
      if (environment_enabled("VISACD_PREPROCESS_TRACE"))
        cerr << "preprocess_cuda_fallback reason=full_output_mismatch\n";
      m = std::move(reference);
      if (metrics)
        *metrics = reference_metrics;
      return;
    }
    if (environment_enabled("VISACD_PREPROCESS_TRACE"))
      cerr << "preprocess_cuda_fallback reason=" << fallback_reason << '\n';
    ManifoldPreprocessMetrics reference_metrics;
    manifold_preprocess_cpu_reference(
        reference, scale, level_set,
        metrics ? &reference_metrics : nullptr);
    m = std::move(reference);
    if (metrics)
      *metrics = reference_metrics;
    return;
  }

  string fallback_reason;
  if (manifold_preprocess_cuda_candidate(m, scale, level_set,
                                         &fallback_reason, metrics))
    return;
  if (environment_enabled("VISACD_PREPROCESS_TRACE"))
    cerr << "preprocess_cuda_fallback reason=" << fallback_reason << '\n';
  manifold_preprocess_cpu_reference(m, scale, level_set, metrics);
}
} // namespace neural_acd
