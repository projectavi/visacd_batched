#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <core.hpp>
#include <cstdlib>
#include <cuda_buffer.hpp>
#include <cuda_runtime.h>
#include <deque>
#include <dlfcn.h>
#include <edge_compaction.hpp>
#include <filesystem>
#include <fstream>
#include <intersections.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <optixUtils.hpp>
#include <optix_function_table_definition.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace neural_acd {
namespace {

void fill_float_array(const vector<Vec3D> &vertices, vector<float> &output) {
  output.resize(vertices.size() * 3);
  for (size_t i = 0; i < vertices.size(); ++i) {
    output[i * 3 + 0] = static_cast<float>(vertices[i][0]);
    output[i * 3 + 1] = static_cast<float>(vertices[i][1]);
    output[i * 3 + 2] = static_cast<float>(vertices[i][2]);
  }
}

string load_ptx(const string &filename) {
  ifstream file(filename, ios::in | ios::binary);
  if (!file)
    throw runtime_error("Failed to open PTX file: " + filename);

  ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

string get_ptx_path() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&load_ptx), &info) && info.dli_fname) {
    auto path =
        filesystem::path(info.dli_fname).parent_path() / "ray_segments.ptx";
    if (filesystem::exists(path))
      return path.string();
  }
  return string(PTX_DIR) + "/ray_segments.ptx";
}

size_t segment_count(size_t n_points) {
  return n_points < 2 ? 0 : n_points * (n_points - 1) / 2;
}

size_t accepted_word_count(size_t segments) {
  return (segments + 31) / 32;
}

void linear_to_pair_batched(long long k, long long n, long long &i,
                            long long &j) {
  const double kd = static_cast<double>(k);
  const double nd = static_cast<double>(n);
  const double t = sqrt(-8.0 * kd + 4.0 * nd * (nd - 1.0) - 7.0);
  i = static_cast<long long>(nd - 2.0 - floor(t / 2.0 - 0.5));
  j = static_cast<long long>(k + i + 1 - n * (n - 1) / 2 +
                             (n - i) * ((n - i) - 1) / 2);
}

size_t estimate_geometry_bytes(const Mesh &mesh) {
  return mesh.vertices.size() * sizeof(float) * 3 +
         mesh.triangles.size() * sizeof(uint3);
}

size_t estimate_request_bytes(const Mesh &points, const Mesh &cage) {
  const size_t segments = segment_count(points.vertices.size());
  const size_t word_bytes =
      accepted_word_count(segments) * sizeof(unsigned int);
  const size_t compaction_bytes =
      segments * sizeof(unsigned int) + word_bytes * 4;
  const size_t point_bytes =
      points.vertices.size() * (sizeof(float) * 3 + sizeof(unsigned int));

  // GAS sizes depend on OptiX and the device. Use a conservative multiplier
  // plus fixed headroom when selecting memory-aware waves.
  return compaction_bytes + point_bytes + estimate_geometry_bytes(cage) * 4 +
         estimate_geometry_bytes(points) * 4 + (8u << 20);
}

OptixBuildFlags configured_build_flags() {
  const char *preference = getenv("VISACD_OPTIX_BUILD_PREFERENCE");
  if (!preference || string(preference) == "trace")
    return OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  if (string(preference) == "build")
    return OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
  if (string(preference) == "none")
    return OPTIX_BUILD_FLAG_NONE;
  throw invalid_argument(
      "VISACD_OPTIX_BUILD_PREFERENCE must be trace, build, or none");
}

size_t configured_max_concurrency() {
  const char *value = getenv("VISACD_OPTIX_MAX_CONCURRENCY");
  if (!value || !*value)
    return 0;

  char *end = nullptr;
  const unsigned long long parsed = strtoull(value, &end, 10);
  if (*end != '\0' || parsed == 0 ||
      parsed > numeric_limits<size_t>::max()) {
    throw invalid_argument(
        "VISACD_OPTIX_MAX_CONCURRENCY must be a positive integer");
  }
  return static_cast<size_t>(parsed);
}

using cuda_memory::DeviceBuffer;

struct OptixSlot {
  cudaStream_t stream = nullptr;
  size_t estimated_request_capacity = 0;
  DeviceBuffer points;
  DeviceBuffer new_mask;
  DeviceBuffer vertices;
  DeviceBuffer indices;
  DeviceBuffer self_indices;
  DeviceBuffer accepted_words;
  DeviceBuffer word_offsets;
  DeviceBuffer compaction_temp;
  DeviceBuffer accepted_count;
  DeviceBuffer compacted_segments;
  DeviceBuffer temp;
  DeviceBuffer gas_output;
  DeviceBuffer self_temp;
  DeviceBuffer self_gas_output;
  DeviceBuffer raygen_record;
  unsigned int *host_accepted_count = nullptr;

  ~OptixSlot() {
    if (stream) {
      cudaStreamSynchronize(stream);
      if (host_accepted_count)
        cudaFreeHost(host_accepted_count);
      cudaStreamDestroy(stream);
    }
  }

  void initialize() {
    CUCHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUCHK(cudaMallocHost(reinterpret_cast<void **>(&host_accepted_count),
                         sizeof(unsigned int)));
  }

  size_t retained_bytes() const {
    return points.capacity() + new_mask.capacity() + vertices.capacity() +
           indices.capacity() + self_indices.capacity() +
           accepted_words.capacity() + word_offsets.capacity() +
           compaction_temp.capacity() + accepted_count.capacity() +
           compacted_segments.capacity() + temp.capacity() +
           gas_output.capacity() + self_temp.capacity() +
           self_gas_output.capacity() + raygen_record.capacity();
  }
};

} // namespace

struct OptixRuntime::Impl {
  OptixDeviceContext context = nullptr;
  OptixModule module = nullptr;
  OptixPipeline pipeline = nullptr;
  OptixProgramGroup raygen_pg = nullptr;
  OptixProgramGroup miss_pg = nullptr;
  OptixProgramGroup hit_pg = nullptr;
  CUdeviceptr d_miss_record = 0;
  CUdeviceptr d_hit_record = 0;
  OptixBuildFlags build_flags = OPTIX_BUILD_FLAG_NONE;
  size_t max_concurrency = 0;
  vector<unique_ptr<OptixSlot>> slots;

  ~Impl() {
    if (context)
      cudaDeviceSynchronize();
    slots.clear();
    if (d_miss_record)
      cudaFree(reinterpret_cast<void *>(d_miss_record));
    if (d_hit_record)
      cudaFree(reinterpret_cast<void *>(d_hit_record));
    if (pipeline)
      optixPipelineDestroy(pipeline);
    if (raygen_pg)
      optixProgramGroupDestroy(raygen_pg);
    if (miss_pg)
      optixProgramGroupDestroy(miss_pg);
    if (hit_pg)
      optixProgramGroupDestroy(hit_pg);
    if (module)
      optixModuleDestroy(module);
    if (context)
      optixDeviceContextDestroy(context);
  }

  void ensure_slots(size_t count) {
    while (slots.size() < count) {
      auto slot = make_unique<OptixSlot>();
      slot->initialize();
      slots.push_back(move(slot));
    }
  }

  void initialize() {
    build_flags = configured_build_flags();
    max_concurrency = configured_max_concurrency();
    context = createContext();
    const string ptx = load_ptx(get_ptx_path());

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 1;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = nullptr;

    char log[2048];
    size_t log_size = sizeof(log);
    OCHK(optixModuleCreate(context, &module_options, &pipeline_options,
                           ptx.c_str(), ptx.size(), log, &log_size, &module));

    OptixProgramGroupOptions group_options = {};
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module;
    raygen_desc.raygen.entryFunctionName = "__raygen__rg";
    log_size = sizeof(log);
    OCHK(optixProgramGroupCreate(context, &raygen_desc, 1, &group_options, log,
                                 &log_size, &raygen_pg));

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module;
    miss_desc.miss.entryFunctionName = "__miss__ms";
    log_size = sizeof(log);
    OCHK(optixProgramGroupCreate(context, &miss_desc, 1, &group_options, log,
                                 &log_size, &miss_pg));

    OptixProgramGroupDesc hit_desc = {};
    hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_desc.hitgroup.moduleAH = module;
    hit_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    log_size = sizeof(log);
    OCHK(optixProgramGroupCreate(context, &hit_desc, 1, &group_options, log,
                                 &log_size, &hit_pg));

    OptixProgramGroup groups[] = {raygen_pg, miss_pg, hit_pg};
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    OCHK(optixPipelineCreate(context, &pipeline_options, &link_options, groups,
                             3, log, &log_size, &pipeline));

    SbtRecord<MissData> miss_record = {};
    OCHK(optixSbtRecordPackHeader(miss_pg, &miss_record));
    SbtRecord<HitgroupData> hit_record = {};
    OCHK(optixSbtRecordPackHeader(hit_pg, &hit_record));

    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_miss_record),
                     sizeof(miss_record)));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_hit_record),
                     sizeof(hit_record)));
    CUCHK(cudaMemcpy(reinterpret_cast<void *>(d_miss_record), &miss_record,
                     sizeof(miss_record), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(reinterpret_cast<void *>(d_hit_record), &hit_record,
                     sizeof(hit_record), cudaMemcpyHostToDevice));
  }
};

OptixRuntime::OptixRuntime() : impl_(make_unique<Impl>()) {
  impl_->initialize();
}
OptixRuntime::~OptixRuntime() = default;
OptixRuntime::OptixRuntime(OptixRuntime &&) noexcept = default;
OptixRuntime &OptixRuntime::operator=(OptixRuntime &&) noexcept = default;

namespace {

struct OptixJob {
  const Mesh &points_mesh;
  const Mesh &target;
  OptixRuntime::Impl &runtime;
  OptixSlot &slot;

  vector<float> h_points;
  vector<unsigned int> h_new_mask;
  vector<float> h_vertices;
  vector<uint3> h_indices;
  vector<uint3> h_self_indices;
  vector<unsigned int> accepted_segments;
  size_t segments = 0;
  size_t word_count = 0;
  size_t scan_temp_bytes = 0;
  unsigned int accepted_count = 0;

  cudaStream_t stream = nullptr;
  float *d_points = nullptr;
  unsigned int *d_new_mask = nullptr;
  float *d_vertices = nullptr;
  uint3 *d_indices = nullptr;
  uint3 *d_self_indices = nullptr;
  unsigned int *d_accepted_words = nullptr;
  unsigned int *d_word_offsets = nullptr;
  void *d_compaction_temp = nullptr;
  unsigned int *d_accepted_count = nullptr;
  unsigned int *d_compacted_segments = nullptr;
  CUdeviceptr d_temp = 0;
  CUdeviceptr d_gas_output = 0;
  CUdeviceptr d_self_temp = 0;
  CUdeviceptr d_self_gas_output = 0;
  CUdeviceptr d_raygen_record = 0;
  OptixTraversableHandle gas = 0;
  OptixTraversableHandle self_gas = 0;
  OptixShaderBindingTable sbt = {};
  OptixAccelBufferSizes gas_sizes = {};
  OptixAccelBufferSizes self_gas_sizes = {};
  OptixBuildInput build_input = {};
  OptixBuildInput self_build_input = {};
  OptixAccelBuildOptions accel_options = {};
  CUdeviceptr d_vertices_ptr = 0;
  CUdeviceptr d_self_vertices_ptr = 0;
  unsigned int triangle_input_flags[1] = {0};

  OptixJob(const Mesh &points_mesh_, const Mesh &target_,
           OptixRuntime::Impl &runtime_, OptixSlot &slot_)
      : points_mesh(points_mesh_), target(target_), runtime(runtime_),
        slot(slot_), stream(slot_.stream) {
    prepare_host_data();
  }

  void prepare_host_data() {
    if (target.vertices.empty() || target.triangles.empty())
      throw invalid_argument("OptiX target mesh must contain triangles");
    if (!points_mesh.is_new.empty() &&
        points_mesh.is_new.size() != points_mesh.vertices.size()) {
      throw invalid_argument("Mesh is_new mask must match its vertices");
    }

    segments = segment_count(points_mesh.vertices.size());
    if (segments > numeric_limits<unsigned int>::max())
      throw overflow_error("Too many segments for one OptiX launch");
    if (target.vertices.size() > numeric_limits<unsigned int>::max() ||
        target.triangles.size() > numeric_limits<unsigned int>::max()) {
      throw overflow_error("OptiX target mesh is too large");
    }

    fill_float_array(points_mesh.vertices, h_points);
    h_new_mask.reserve(points_mesh.is_new.size());
    for (bool is_new : points_mesh.is_new)
      h_new_mask.push_back(is_new ? 1u : 0u);
    fill_float_array(target.vertices, h_vertices);
    h_indices.reserve(target.triangles.size());
    for (const auto &triangle : target.triangles) {
      h_indices.push_back(make_uint3(static_cast<unsigned>(triangle[0]),
                                     static_cast<unsigned>(triangle[1]),
                                     static_cast<unsigned>(triangle[2])));
    }
    h_self_indices.reserve(points_mesh.triangles.size());
    for (const auto &triangle : points_mesh.triangles) {
      h_self_indices.push_back(
          make_uint3(static_cast<unsigned>(triangle[0]),
                     static_cast<unsigned>(triangle[1]),
                     static_cast<unsigned>(triangle[2])));
    }
    word_count = accepted_word_count(segments);
  }

  void allocate() {
    if (word_count == 0)
      return;

    DeviceBuffer::set_allocation_stream(stream);

    slot.points.ensure(sizeof(float) * h_points.size());
    d_points = slot.points.as<float>();
    if (!h_new_mask.empty()) {
      slot.new_mask.ensure(sizeof(unsigned int) * h_new_mask.size());
      d_new_mask = slot.new_mask.as<unsigned int>();
    }
    slot.vertices.ensure(sizeof(float) * h_vertices.size());
    d_vertices = slot.vertices.as<float>();
    slot.indices.ensure(sizeof(uint3) * h_indices.size());
    d_indices = slot.indices.as<uint3>();
    slot.self_indices.ensure(sizeof(uint3) * h_self_indices.size());
    d_self_indices = slot.self_indices.as<uint3>();
    slot.accepted_words.ensure(sizeof(unsigned int) * word_count);
    d_accepted_words = slot.accepted_words.as<unsigned int>();
    slot.word_offsets.ensure(sizeof(unsigned int) * word_count);
    d_word_offsets = slot.word_offsets.as<unsigned int>();
    scan_temp_bytes = edge_compaction_temp_bytes(word_count);
    slot.compaction_temp.ensure(scan_temp_bytes);
    d_compaction_temp = slot.compaction_temp.as<void>();
    slot.accepted_count.ensure(sizeof(unsigned int));
    d_accepted_count = slot.accepted_count.as<unsigned int>();
    slot.raygen_record.ensure(sizeof(SbtRecord<RayGenData>));
    d_raygen_record = slot.raygen_record.device_ptr();

    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    d_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_vertices);
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.numVertices =
        static_cast<unsigned int>(target.vertices.size());
    build_input.triangleArray.vertexBuffers = &d_vertices_ptr;
    build_input.triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets =
        static_cast<unsigned int>(target.triangles.size());
    build_input.triangleArray.indexBuffer =
        reinterpret_cast<CUdeviceptr>(d_indices);
    build_input.triangleArray.flags = triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;

    self_build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    d_self_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_points);
    self_build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    self_build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    self_build_input.triangleArray.numVertices =
        static_cast<unsigned int>(points_mesh.vertices.size());
    self_build_input.triangleArray.vertexBuffers = &d_self_vertices_ptr;
    self_build_input.triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    self_build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    self_build_input.triangleArray.numIndexTriplets =
        static_cast<unsigned int>(points_mesh.triangles.size());
    self_build_input.triangleArray.indexBuffer =
        reinterpret_cast<CUdeviceptr>(d_self_indices);
    self_build_input.triangleArray.flags = triangle_input_flags;
    self_build_input.triangleArray.numSbtRecords = 1;

    accel_options.buildFlags = runtime.build_flags;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OCHK(optixAccelComputeMemoryUsage(runtime.context, &accel_options,
                                      &build_input, 1, &gas_sizes));
    OCHK(optixAccelComputeMemoryUsage(runtime.context, &accel_options,
                                      &self_build_input, 1,
                                      &self_gas_sizes));
    slot.temp.ensure(gas_sizes.tempSizeInBytes);
    d_temp = slot.temp.device_ptr();
    slot.gas_output.ensure(gas_sizes.outputSizeInBytes);
    d_gas_output = slot.gas_output.device_ptr();
    slot.self_temp.ensure(self_gas_sizes.tempSizeInBytes);
    d_self_temp = slot.self_temp.device_ptr();
    slot.self_gas_output.ensure(self_gas_sizes.outputSizeInBytes);
    d_self_gas_output = slot.self_gas_output.device_ptr();
  }

  void enqueue() {
    if (word_count == 0)
      return;

    CUCHK(cudaMemcpyAsync(d_points, h_points.data(),
                          sizeof(float) * h_points.size(),
                          cudaMemcpyHostToDevice, stream));
    if (!h_new_mask.empty()) {
      CUCHK(cudaMemcpyAsync(d_new_mask, h_new_mask.data(),
                            sizeof(unsigned int) * h_new_mask.size(),
                            cudaMemcpyHostToDevice, stream));
    }
    CUCHK(cudaMemcpyAsync(d_vertices, h_vertices.data(),
                          sizeof(float) * h_vertices.size(),
                          cudaMemcpyHostToDevice, stream));
    CUCHK(cudaMemcpyAsync(d_indices, h_indices.data(),
                          sizeof(uint3) * h_indices.size(),
                          cudaMemcpyHostToDevice, stream));
    CUCHK(cudaMemcpyAsync(d_self_indices, h_self_indices.data(),
                          sizeof(uint3) * h_self_indices.size(),
                          cudaMemcpyHostToDevice, stream));
    CUCHK(cudaMemsetAsync(d_accepted_words, 0,
                          sizeof(unsigned int) * word_count, stream));

    OCHK(optixAccelBuild(runtime.context, stream, &accel_options, &build_input,
                         1, d_temp, gas_sizes.tempSizeInBytes, d_gas_output,
                         gas_sizes.outputSizeInBytes, &gas, nullptr, 0));
    OCHK(optixAccelBuild(runtime.context, stream, &accel_options,
                         &self_build_input, 1, d_self_temp,
                         self_gas_sizes.tempSizeInBytes, d_self_gas_output,
                         self_gas_sizes.outputSizeInBytes, &self_gas, nullptr,
                         0));

    SbtRecord<RayGenData> raygen_record = {};
    OCHK(optixSbtRecordPackHeader(runtime.raygen_pg, &raygen_record));
    raygen_record.data.points = d_points;
    raygen_record.data.new_mask = d_new_mask;
    raygen_record.data.n_points =
        static_cast<long long>(points_mesh.vertices.size());
    raygen_record.data.has_mask = h_new_mask.empty() ? 0u : 1u;
    raygen_record.data.accepted_words = d_accepted_words;
    raygen_record.data.cage_gas = gas;
    raygen_record.data.self_gas = self_gas;
    CUCHK(cudaMemcpyAsync(reinterpret_cast<void *>(d_raygen_record),
                          &raygen_record, sizeof(raygen_record),
                          cudaMemcpyHostToDevice, stream));

    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = runtime.d_miss_record;
    sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = runtime.d_hit_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitgroupData>);
    sbt.hitgroupRecordCount = 1;

    OCHK(optixLaunch(runtime.pipeline, stream, 0, 0, &sbt,
                     static_cast<unsigned int>(segments), 1, 1));
  }

  void enqueue_count() {
    if (word_count == 0)
      return;
    count_compacted_segments_async(
        d_accepted_words, d_word_offsets, word_count, d_accepted_count,
        d_compaction_temp, scan_temp_bytes, stream);
    CUCHK(cudaMemcpyAsync(slot.host_accepted_count, d_accepted_count,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost,
                          stream));
  }

  void finish_count() {
    if (word_count == 0)
      return;
    accepted_count = *slot.host_accepted_count;
    accepted_segments.resize(accepted_count);
  }

  void allocate_compacted_segments() {
    if (accepted_count == 0)
      return;
    DeviceBuffer::set_allocation_stream(stream);
    slot.compacted_segments.ensure(sizeof(unsigned int) * accepted_count);
    d_compacted_segments = slot.compacted_segments.as<unsigned int>();
  }

  void enqueue_compacted_segments() {
    if (accepted_count == 0)
      return;
    scatter_compacted_segments_async(
        d_accepted_words, d_word_offsets, word_count, segments,
        d_compacted_segments, stream);
    CUCHK(cudaMemcpyAsync(accepted_segments.data(), d_compacted_segments,
                          sizeof(unsigned int) * accepted_count,
                          cudaMemcpyDeviceToHost, stream));
  }

};

class CompletionQueue {
public:
  void notify(size_t index) noexcept {
    {
      lock_guard<mutex> lock(mutex_);
      ready_.push_back(index);
    }
    condition_.notify_one();
  }

  size_t wait() {
    unique_lock<mutex> lock(mutex_);
    condition_.wait(lock, [this]() { return !ready_.empty(); });
    const size_t index = ready_.front();
    ready_.pop_front();
    return index;
  }

private:
  mutex mutex_;
  condition_variable condition_;
  deque<size_t> ready_;
};

struct CompletionPayload {
  CompletionQueue *queue = nullptr;
  size_t index = 0;
};

void CUDART_CB notify_completion(void *payload) {
  auto *completion = static_cast<CompletionPayload *>(payload);
  completion->queue->notify(completion->index);
}

vector<vector<pair<unsigned int, unsigned int>>>
run_wave(const vector<pair<Mesh *, Mesh *>> &requests, size_t begin, size_t end,
         OptixRuntime::Impl &runtime, BatchExecutor *executor) {
  runtime.ensure_slots(end - begin);
  vector<unique_ptr<OptixJob>> jobs(end - begin);
  const auto prepare_job = [&](size_t local_idx) {
    const size_t request_idx = begin + local_idx;
    jobs[local_idx] = make_unique<OptixJob>(
        *requests[request_idx].first, *requests[request_idx].second, runtime,
        *runtime.slots[local_idx]);
  };
  if (executor)
    executor->parallel_for_priority(jobs.size(), prepare_job);
  else
    for (size_t local_idx = 0; local_idx < jobs.size(); ++local_idx)
      prepare_job(local_idx);

  // Grow runtime-owned buffers before submitting the first job. Later waves
  // and decomposition iterations reuse both these buffers and their streams.
  for (auto &job : jobs)
    job->allocate();
  for (auto &job : jobs)
    job->enqueue();

  CompletionQueue count_completions;
  CompletionQueue result_completions;
  vector<CompletionPayload> count_payloads(jobs.size());
  vector<CompletionPayload> result_payloads(jobs.size());
  for (size_t local_idx = 0; local_idx < jobs.size(); ++local_idx) {
    jobs[local_idx]->enqueue_count();
    count_payloads[local_idx] = {&count_completions, local_idx};
    CUCHK(cudaLaunchHostFunc(jobs[local_idx]->stream, notify_completion,
                            &count_payloads[local_idx]));
  }

  for (size_t completed = 0; completed < jobs.size(); ++completed) {
    const size_t local_idx = count_completions.wait();
    OptixJob &job = *jobs[local_idx];
    job.finish_count();
    job.allocate_compacted_segments();
    runtime.slots[local_idx]->estimated_request_capacity =
        runtime.slots[local_idx]->retained_bytes();
    job.enqueue_compacted_segments();
    result_payloads[local_idx] = {&result_completions, local_idx};
    CUCHK(cudaLaunchHostFunc(job.stream, notify_completion,
                            &result_payloads[local_idx]));
  }
  for (size_t completed = 0; completed < jobs.size(); ++completed)
    result_completions.wait();

  vector<vector<pair<unsigned int, unsigned int>>> results(end - begin);
  const auto decode_result = [&](size_t local_idx) {
    const Mesh &mesh = *requests[begin + local_idx].first;
    const vector<unsigned int> &accepted_segments =
        jobs[local_idx]->accepted_segments;
    auto &edges = results[local_idx];
    edges.reserve(accepted_segments.size());

    for (unsigned int segment : accepted_segments) {
      long long first, second;
      linear_to_pair_batched(static_cast<long long>(segment),
                             static_cast<long long>(mesh.vertices.size()),
                             first, second);
      edges.emplace_back(static_cast<unsigned int>(first),
                         static_cast<unsigned int>(second));
    }
  };
  if (executor)
    executor->parallel_for_priority(results.size(), decode_result);
  else
    for (size_t local_idx = 0; local_idx < results.size(); ++local_idx)
      decode_result(local_idx);
  return results;
}

} // namespace

vector<vector<pair<unsigned int, unsigned int>>>
compute_intersection_matrices(
    const vector<pair<Mesh *, Mesh *>> &requests, OptixRuntime &runtime,
    size_t max_batch_size, double memory_fraction, BatchExecutor *executor) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0)
    throw invalid_argument("batch_memory_fraction must be in (0, 1]");
  for (const auto &request : requests) {
    if (!request.first || !request.second)
      throw invalid_argument("Intersection requests cannot contain null meshes");
  }

  vector<vector<pair<unsigned int, unsigned int>>> results(requests.size());
  size_t begin = 0;
  while (begin < requests.size()) {
    size_t free_bytes = 0, total_bytes = 0;
    CUCHK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);

    size_t end = begin;
    size_t additional_bytes = 0;
    while (end < requests.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      if (runtime.impl_->max_concurrency &&
          end - begin >= runtime.impl_->max_concurrency)
        break;
      const Mesh &mesh = *requests[end].first;
      const size_t request_bytes =
          estimate_request_bytes(mesh, *requests[end].second);
      const size_t slot_index = end - begin;
      const size_t retained_bytes =
          slot_index < runtime.impl_->slots.size()
              ? runtime.impl_->slots[slot_index]->estimated_request_capacity
              : 0;
      const size_t request_growth =
          request_bytes > retained_bytes ? request_bytes - retained_bytes : 0;
      if (end > begin && additional_bytes + request_growth > budget)
        break;
      additional_bytes += request_growth;
      ++end;
    }
    if (end == begin)
      ++end;

    auto wave_results =
        run_wave(requests, begin, end, *runtime.impl_, executor);
    for (size_t i = 0; i < wave_results.size(); ++i)
      results[begin + i] = move(wave_results[i]);
    begin = end;
  }

  return results;
}

} // namespace neural_acd
