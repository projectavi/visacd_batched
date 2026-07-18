#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace neural_acd::cuda_memory {

inline void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

inline bool async_pool_available() {
  static std::once_flag initialized;
  static bool available = false;
  std::call_once(initialized, []() {
    const char *disabled = std::getenv("VISACD_DISABLE_ASYNC_ALLOCATOR");
    if (disabled && *disabled && std::string(disabled) != "0")
      return;
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported,
                               0) != cudaSuccess ||
        !supported) {
      cudaGetLastError();
      return;
    }
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) != cudaSuccess) {
      cudaGetLastError();
      return;
    }
    unsigned long long threshold =
        std::numeric_limits<unsigned long long>::max();
    if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                &threshold) != cudaSuccess) {
      cudaGetLastError();
      return;
    }
    available = true;
  });
  return available;
}

class DeviceBuffer {
public:
  ~DeviceBuffer() {
    if (data_)
      cudaFree(data_);
  }

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  static void set_allocation_stream(cudaStream_t stream) {
    allocation_stream_ = stream;
  }

  void ensure(size_t bytes, const char *operation = "allocate CUDA buffer") {
    if (bytes <= capacity_)
      return;
    void *replacement = nullptr;
    const bool use_async = allocation_stream_ && async_pool_available();
    if (use_async)
      check(cudaMallocAsync(&replacement, bytes, allocation_stream_),
            operation);
    else
      check(cudaMalloc(&replacement, bytes), operation);

    if (data_) {
      if (use_async)
        check(cudaFreeAsync(data_, allocation_stream_),
              "release pooled CUDA buffer");
      else
        check(cudaFree(data_), "release CUDA buffer");
    }
    data_ = replacement;
    capacity_ = bytes;
  }

  template <typename T> T *as() const { return static_cast<T *>(data_); }
  size_t capacity() const { return capacity_; }
  CUdeviceptr device_ptr() const {
    return reinterpret_cast<CUdeviceptr>(data_);
  }

private:
  void *data_ = nullptr;
  size_t capacity_ = 0;
  inline static thread_local cudaStream_t allocation_stream_ = nullptr;
};

class PinnedBuffer {
public:
  ~PinnedBuffer() {
    if (data_)
      cudaFreeHost(data_);
  }

  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  void ensure(size_t bytes,
              const char *operation = "allocate pinned CUDA buffer") {
    if (bytes <= capacity_)
      return;
    void *replacement = nullptr;
    check(cudaMallocHost(&replacement, bytes), operation);
    if (data_)
      check(cudaFreeHost(data_), "release pinned CUDA buffer");
    data_ = replacement;
    capacity_ = bytes;
  }

  template <typename T> T *as() const { return static_cast<T *>(data_); }
  size_t capacity() const { return capacity_; }

private:
  void *data_ = nullptr;
  size_t capacity_ = 0;
};

} // namespace neural_acd::cuda_memory
