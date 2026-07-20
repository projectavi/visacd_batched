#pragma once

#include <memory>
#include <mutex>

namespace neural_acd {

template <typename Runtime> class ResettableRuntime {
public:
  Runtime &get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!runtime_)
      runtime_ = std::make_unique<Runtime>();
    return *runtime_;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    runtime_.reset();
  }

  ResettableRuntime() = default;
  ResettableRuntime(const ResettableRuntime &) = delete;
  ResettableRuntime &operator=(const ResettableRuntime &) = delete;

private:
  std::mutex mutex_;
  std::unique_ptr<Runtime> runtime_;
};

} // namespace neural_acd
