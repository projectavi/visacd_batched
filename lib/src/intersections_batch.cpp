#include <cmath>
#include <core.hpp>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <intersections.hpp>
#include <limits>
#include <memory>
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
  const size_t pair_bytes =
      accepted_word_count(segment_count(points.vertices.size())) *
      sizeof(unsigned int);
  const size_t point_bytes =
      points.vertices.size() * (sizeof(float) * 3 + sizeof(unsigned int));

  // GAS sizes depend on OptiX and the device. This conservative allowance
  // keeps the quadratic result array as the dominant part of the estimate.
  return pair_bytes + point_bytes + estimate_geometry_bytes(cage) * 4 +
         estimate_geometry_bytes(points) * 4 + (8u << 20);
}

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

  ~Impl() {
    if (context)
      cudaDeviceSynchronize();
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

  void initialize() {
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

  vector<float> h_points;
  vector<unsigned int> h_new_mask;
  vector<float> h_vertices;
  vector<uint3> h_indices;
  vector<uint3> h_self_indices;
  vector<unsigned int> accepted_words;

  cudaStream_t stream = nullptr;
  float *d_points = nullptr;
  unsigned int *d_new_mask = nullptr;
  float *d_vertices = nullptr;
  uint3 *d_indices = nullptr;
  uint3 *d_self_indices = nullptr;
  unsigned int *d_accepted_words = nullptr;
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
           OptixRuntime::Impl &runtime_)
      : points_mesh(points_mesh_), target(target_), runtime(runtime_) {
    prepare_host_data();
  }

  ~OptixJob() {
    if (stream)
      cudaStreamSynchronize(stream);
    if (d_points)
      cudaFree(d_points);
    if (d_new_mask)
      cudaFree(d_new_mask);
    if (d_vertices)
      cudaFree(d_vertices);
    if (d_indices)
      cudaFree(d_indices);
    if (d_self_indices)
      cudaFree(d_self_indices);
    if (d_accepted_words)
      cudaFree(d_accepted_words);
    if (d_temp)
      cudaFree(reinterpret_cast<void *>(d_temp));
    if (d_gas_output)
      cudaFree(reinterpret_cast<void *>(d_gas_output));
    if (d_self_temp)
      cudaFree(reinterpret_cast<void *>(d_self_temp));
    if (d_self_gas_output)
      cudaFree(reinterpret_cast<void *>(d_self_gas_output));
    if (d_raygen_record)
      cudaFree(reinterpret_cast<void *>(d_raygen_record));
    if (stream)
      cudaStreamDestroy(stream);
  }

  void prepare_host_data() {
    if (target.vertices.empty() || target.triangles.empty())
      throw invalid_argument("OptiX target mesh must contain triangles");
    if (!points_mesh.is_new.empty() &&
        points_mesh.is_new.size() != points_mesh.vertices.size()) {
      throw invalid_argument("Mesh is_new mask must match its vertices");
    }

    const size_t segments = segment_count(points_mesh.vertices.size());
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
    accepted_words.resize(accepted_word_count(segments));
  }

  void allocate() {
    if (accepted_words.empty())
      return;

    CUCHK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_points),
                     sizeof(float) * h_points.size()));
    if (!h_new_mask.empty()) {
      CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_new_mask),
                       sizeof(unsigned int) * h_new_mask.size()));
    }
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_vertices),
                     sizeof(float) * h_vertices.size()));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_indices),
                     sizeof(uint3) * h_indices.size()));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_self_indices),
                     sizeof(uint3) * h_self_indices.size()));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_accepted_words),
                     sizeof(unsigned int) * accepted_words.size()));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_record),
                     sizeof(SbtRecord<RayGenData>)));

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

    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OCHK(optixAccelComputeMemoryUsage(runtime.context, &accel_options,
                                      &build_input, 1, &gas_sizes));
    OCHK(optixAccelComputeMemoryUsage(runtime.context, &accel_options,
                                      &self_build_input, 1,
                                      &self_gas_sizes));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_temp),
                     gas_sizes.tempSizeInBytes));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output),
                     gas_sizes.outputSizeInBytes));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_self_temp),
                     self_gas_sizes.tempSizeInBytes));
    CUCHK(cudaMalloc(reinterpret_cast<void **>(&d_self_gas_output),
                     self_gas_sizes.outputSizeInBytes));
  }

  void enqueue() {
    if (accepted_words.empty())
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
                          sizeof(unsigned int) * accepted_words.size(),
                          stream));

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
                     static_cast<unsigned int>(
                         segment_count(points_mesh.vertices.size())),
                     1, 1));
  }

  void download() {
    if (accepted_words.empty())
      return;
    CUCHK(cudaMemcpyAsync(accepted_words.data(), d_accepted_words,
                          sizeof(unsigned int) * accepted_words.size(),
                          cudaMemcpyDeviceToHost, stream));
  }

  void wait() {
    if (stream)
      CUCHK(cudaStreamSynchronize(stream));
  }
};

vector<vector<pair<unsigned int, unsigned int>>>
run_wave(const vector<pair<Mesh *, Mesh *>> &requests, size_t begin, size_t end,
         OptixRuntime::Impl &runtime) {
  vector<unique_ptr<OptixJob>> jobs;
  jobs.reserve(end - begin);
  for (size_t i = begin; i < end; ++i) {
    jobs.push_back(
        make_unique<OptixJob>(*requests[i].first, *requests[i].second, runtime));
  }

  // cudaMalloc may synchronize the device, so finish every allocation before
  // submitting the first job in a wave.
  for (auto &job : jobs)
    job->allocate();
  for (auto &job : jobs)
    job->enqueue();
  for (auto &job : jobs)
    job->download();
  for (auto &job : jobs)
    job->wait();

  vector<vector<pair<unsigned int, unsigned int>>> results(end - begin);
  for (size_t local_idx = 0; local_idx < end - begin; ++local_idx) {
    const Mesh &mesh = *requests[begin + local_idx].first;
    const vector<unsigned int> &accepted_words =
        jobs[local_idx]->accepted_words;
    const size_t segments = segment_count(mesh.vertices.size());
    auto &edges = results[local_idx];

    for (size_t word_index = 0; word_index < accepted_words.size();
         ++word_index) {
      unsigned int word = accepted_words[word_index];
      while (word) {
        const unsigned int bit =
            static_cast<unsigned int>(__builtin_ctz(word));
        const size_t segment = word_index * 32 + bit;
        if (segment < segments) {
          long long first, second;
          linear_to_pair_batched(static_cast<long long>(segment),
                                 static_cast<long long>(mesh.vertices.size()),
                                 first, second);
          edges.emplace_back(static_cast<unsigned int>(first),
                             static_cast<unsigned int>(second));
        }
        word &= word - 1;
      }
    }
  }
  return results;
}

} // namespace

vector<vector<pair<unsigned int, unsigned int>>>
compute_intersection_matrices(
    const vector<pair<Mesh *, Mesh *>> &requests, OptixRuntime &runtime,
    size_t max_batch_size, double memory_fraction) {
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
    size_t estimated_bytes = 0;
    while (end < requests.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const Mesh &mesh = *requests[end].first;
      const size_t request_bytes =
          estimate_request_bytes(mesh, *requests[end].second);
      if (end > begin && estimated_bytes + request_bytes > budget)
        break;
      estimated_bytes += request_bytes;
      ++end;
    }
    if (end == begin)
      ++end;

    auto wave_results = run_wave(requests, begin, end, *runtime.impl_);
    for (size_t i = 0; i < wave_results.size(); ++i)
      results[begin + i] = move(wave_results[i]);
    begin = end;
  }

  return results;
}

} // namespace neural_acd
