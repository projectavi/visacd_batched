#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace neural_acd {

size_t edge_compaction_temp_bytes(size_t word_count);

void count_compacted_segments_async(
    const unsigned int *accepted_words, unsigned int *word_offsets,
    size_t word_count, unsigned int *accepted_count, void *temporary_storage,
    size_t temporary_bytes, cudaStream_t stream);

void scatter_compacted_segments_async(
    const unsigned int *accepted_words, const unsigned int *word_offsets,
    size_t word_count, size_t segment_count, unsigned int *segments,
    cudaStream_t stream);

} // namespace neural_acd
