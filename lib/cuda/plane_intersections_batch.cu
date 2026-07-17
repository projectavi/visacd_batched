#include <cuda_runtime.h>
#include <memory>
#include <plane_intersections.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace neural_acd {
namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

__global__ void plane_score_batch_kernel(
    const float *planes, const float *points, const unsigned int *edges,
    float *scores, int num_planes, int num_points, int num_edges,
    float eps = 1e-6f) {
  const int plane_idx = blockIdx.x;
  if (plane_idx >= num_planes)
    return;

  const float4 plane = reinterpret_cast<const float4 *>(planes)[plane_idx];
  float thread_score = 0.0f;
  for (int edge_idx = threadIdx.x; edge_idx < num_edges;
       edge_idx += blockDim.x) {
    const unsigned int first = edges[edge_idx * 2];
    const unsigned int second = edges[edge_idx * 2 + 1];
    if (first >= static_cast<unsigned int>(num_points) ||
        second >= static_cast<unsigned int>(num_points)) {
      continue;
    }

    const float3 p1 = reinterpret_cast<const float3 *>(points)[first];
    const float3 p2 = reinterpret_cast<const float3 *>(points)[second];
    const float value1 =
        plane.x * p1.x + plane.y * p1.y + plane.z * p1.z + plane.w;
    const float value2 =
        plane.x * p2.x + plane.y * p2.y + plane.z * p2.z + plane.w;
    const int side1 = (value1 > eps) - (value1 < -eps);
    const int side2 = (value2 > eps) - (value2 < -eps);
    if (side1 == 0 || side2 == 0 || side1 == side2)
      continue;

    const float dx = p2.x - p1.x;
    const float dy = p2.y - p1.y;
    const float dz = p2.z - p1.z;
    thread_score += sqrtf(dx * dx + dy * dy + dz * dz);
  }

  __shared__ float partial_scores[256];
  partial_scores[threadIdx.x] = thread_score;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial_scores[threadIdx.x] +=
          partial_scores[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scores[plane_idx] = partial_scores[0];
}

size_t estimate_bytes(const PlaneScoreInput &input) {
  return sizeof(float) * 4 * input.num_planes +
         sizeof(float) * 3 * input.num_points +
         sizeof(unsigned int) * 2 * input.num_edges +
         sizeof(float) * input.num_planes;
}

struct PlaneScoreJob {
  const PlaneScoreInput &input;
  cudaStream_t stream = nullptr;
  float *d_planes = nullptr;
  float *d_points = nullptr;
  unsigned int *d_edges = nullptr;
  float *d_scores = nullptr;

  explicit PlaneScoreJob(const PlaneScoreInput &input_) : input(input_) {}

  ~PlaneScoreJob() {
    if (stream)
      cudaStreamSynchronize(stream);
    if (d_planes)
      cudaFree(d_planes);
    if (d_points)
      cudaFree(d_points);
    if (d_edges)
      cudaFree(d_edges);
    if (d_scores)
      cudaFree(d_scores);
    if (stream)
      cudaStreamDestroy(stream);
  }

  void allocate() {
    if (!input.num_planes || !input.num_edges)
      return;

    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_planes),
                          sizeof(float) * 4 * input.num_planes),
               "cudaMalloc planes");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_points),
                          sizeof(float) * 3 * input.num_points),
               "cudaMalloc points");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_edges),
                          sizeof(unsigned int) * 2 * input.num_edges),
               "cudaMalloc edges");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_scores),
                          sizeof(float) * input.num_planes),
               "cudaMalloc scores");
  }

  void enqueue() {
    if (!stream)
      return;

    check_cuda(cudaMemcpyAsync(d_planes, input.planes,
                               sizeof(float) * 4 * input.num_planes,
                               cudaMemcpyHostToDevice, stream),
               "copy planes");
    check_cuda(cudaMemcpyAsync(d_points, input.points,
                               sizeof(float) * 3 * input.num_points,
                               cudaMemcpyHostToDevice, stream),
               "copy points");
    check_cuda(cudaMemcpyAsync(d_edges, input.edges,
                               sizeof(unsigned int) * 2 * input.num_edges,
                               cudaMemcpyHostToDevice, stream),
               "copy edges");
    constexpr int block_size = 256;
    plane_score_batch_kernel<<<input.num_planes, block_size, 0, stream>>>(
        d_planes, d_points, d_edges, d_scores, input.num_planes,
        input.num_points, input.num_edges);
    check_cuda(cudaGetLastError(), "plane scoring kernel launch");
  }

  void download() {
    if (!stream)
      return;
    check_cuda(cudaMemcpyAsync(input.scores, d_scores,
                               sizeof(float) * input.num_planes,
                               cudaMemcpyDeviceToHost, stream),
               "copy plane scores");
  }

  void wait() {
    if (stream)
      check_cuda(cudaStreamSynchronize(stream), "plane scoring");
  }
};

void run_wave(const std::vector<PlaneScoreInput> &inputs, size_t begin,
              size_t end) {
  std::vector<std::unique_ptr<PlaneScoreJob>> jobs;
  jobs.reserve(end - begin);
  for (size_t i = begin; i < end; ++i)
    jobs.push_back(std::make_unique<PlaneScoreJob>(inputs[i]));

  // Avoid cudaMalloc serialization after any kernels have been submitted.
  for (auto &job : jobs)
    job->allocate();
  for (auto &job : jobs)
    job->enqueue();
  for (auto &job : jobs)
    job->download();
  for (auto &job : jobs)
    job->wait();
}

} // namespace

void classify_and_rate_planes_batch(
    const std::vector<PlaneScoreInput> &inputs, size_t max_batch_size,
    double memory_fraction) {
  if (memory_fraction <= 0.0 || memory_fraction > 1.0)
    throw std::invalid_argument("batch_memory_fraction must be in (0, 1]");

  for (const auto &input : inputs) {
    if (input.num_planes < 0 || input.num_points < 0 || input.num_edges < 0)
      throw std::invalid_argument("Plane scoring counts cannot be negative");
    if ((input.num_planes && (!input.planes || !input.scores)) ||
        (input.num_points && !input.points) ||
        (input.num_edges && !input.edges)) {
      throw std::invalid_argument("Plane scoring input contains a null buffer");
    }
  }

  size_t begin = 0;
  while (begin < inputs.size()) {
    size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const size_t budget =
        static_cast<size_t>(static_cast<double>(free_bytes) * memory_fraction);

    size_t end = begin;
    size_t bytes = 0;
    while (end < inputs.size()) {
      if (max_batch_size && end - begin >= max_batch_size)
        break;
      const size_t next_bytes = estimate_bytes(inputs[end]);
      if (end > begin && bytes + next_bytes > budget)
        break;
      bytes += next_bytes;
      ++end;
    }
    if (end == begin)
      ++end;

    run_wave(inputs, begin, end);
    begin = end;
  }
}

} // namespace neural_acd
