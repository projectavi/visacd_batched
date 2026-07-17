#include <cub/cub.cuh>
#include <edge_compaction.hpp>
#include <limits>
#include <stdexcept>
#include <string>

namespace neural_acd {
namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

int checked_word_count(size_t word_count) {
  if (word_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("Edge bitset exceeds CUDA scan limit");
  return static_cast<int>(word_count);
}

struct Popcount {
  __host__ __device__ unsigned int operator()(unsigned int word) const {
#ifdef __CUDA_ARCH__
    return __popc(word);
#else
    return static_cast<unsigned int>(__builtin_popcount(word));
#endif
  }
};

using PopcountIterator =
    cub::TransformInputIterator<unsigned int, Popcount,
                                const unsigned int *>;

__global__ void finalize_count_kernel(const unsigned int *accepted_words,
                                      const unsigned int *word_offsets,
                                      int word_count,
                                      unsigned int *accepted_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  if (word_count == 0) {
    *accepted_count = 0;
    return;
  }
  const int last = word_count - 1;
  *accepted_count = word_offsets[last] + __popc(accepted_words[last]);
}

__global__ void scatter_segments_kernel(
    const unsigned int *accepted_words, const unsigned int *word_offsets,
    int word_count, unsigned int segment_count, unsigned int *segments) {
  const int word_index =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (word_index >= word_count)
    return;

  unsigned int word = accepted_words[word_index];
  unsigned int output_index = word_offsets[word_index];
  while (word) {
    const unsigned int bit = static_cast<unsigned int>(__ffs(word) - 1);
    const unsigned int segment =
        static_cast<unsigned int>(word_index) * 32u + bit;
    if (segment < segment_count)
      segments[output_index++] = segment;
    word &= word - 1;
  }
}

} // namespace

size_t edge_compaction_temp_bytes(size_t word_count) {
  const int count = checked_word_count(word_count);
  size_t temporary_bytes = 0;
  PopcountIterator counts(nullptr, Popcount{});
  check_cuda(cub::DeviceScan::ExclusiveSum(
                 nullptr, temporary_bytes, counts,
                 static_cast<unsigned int *>(nullptr), count),
             "query edge compaction scan storage");
  return temporary_bytes;
}

void count_compacted_segments_async(
    const unsigned int *accepted_words, unsigned int *word_offsets,
    size_t word_count, unsigned int *accepted_count, void *temporary_storage,
    size_t temporary_bytes, cudaStream_t stream) {
  const int count = checked_word_count(word_count);
  if (count == 0) {
    check_cuda(cudaMemsetAsync(accepted_count, 0, sizeof(unsigned int), stream),
               "clear compacted edge count");
    return;
  }

  PopcountIterator counts(accepted_words, Popcount{});
  check_cuda(cub::DeviceScan::ExclusiveSum(
                 temporary_storage, temporary_bytes, counts, word_offsets,
                 count, stream),
             "scan accepted edge words");
  finalize_count_kernel<<<1, 1, 0, stream>>>(
      accepted_words, word_offsets, count, accepted_count);
  check_cuda(cudaGetLastError(), "finalize compacted edge count");
}

void scatter_compacted_segments_async(
    const unsigned int *accepted_words, const unsigned int *word_offsets,
    size_t word_count, size_t segment_count, unsigned int *segments,
    cudaStream_t stream) {
  const int count = checked_word_count(word_count);
  if (count == 0 || segment_count == 0)
    return;
  if (segment_count >
      static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::overflow_error("Segment count exceeds CUDA indexing limit");
  }

  constexpr int block_size = 256;
  const int block_count = (count + block_size - 1) / block_size;
  scatter_segments_kernel<<<block_count, block_size, 0, stream>>>(
      accepted_words, word_offsets, count,
      static_cast<unsigned int>(segment_count), segments);
  check_cuda(cudaGetLastError(), "scatter compacted edge segments");
}

} // namespace neural_acd
