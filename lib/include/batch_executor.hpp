#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace neural_acd {

class BatchExecutor {
public:
  explicit BatchExecutor(size_t thread_count)
      : thread_count_(std::max<size_t>(1, thread_count)) {
    workers_.reserve(thread_count_);
    for (size_t i = 0; i < thread_count_; ++i)
      workers_.emplace_back([this]() { worker_loop(); });
  }

  ~BatchExecutor() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stopping_ = true;
    }
    task_condition_.notify_all();
    for (std::thread &worker : workers_)
      worker.join();
  }

  BatchExecutor(const BatchExecutor &) = delete;
  BatchExecutor &operator=(const BatchExecutor &) = delete;

  size_t thread_count() const { return thread_count_; }

  void submit(std::function<void()> task) { enqueue(std::move(task), false); }

  void submit_priority(std::function<void()> task) {
    enqueue(std::move(task), true);
  }

  template <typename Function>
  void parallel_for(size_t work_size, Function function) {
    parallel_for_impl(work_size, std::move(function), false);
  }

  template <typename Function>
  void parallel_for_priority(size_t work_size, Function function) {
    parallel_for_impl(work_size, std::move(function), true);
  }

private:
  void enqueue(std::function<void()> task, bool priority) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (stopping_)
        throw std::runtime_error("Cannot submit work to a stopped executor");
      if (priority)
        tasks_.push_front(std::move(task));
      else
        tasks_.push_back(std::move(task));
    }
    task_condition_.notify_one();
  }

  template <typename Function>
  void parallel_for_impl(size_t work_size, Function function, bool priority) {
    if (work_size == 0)
      return;

    auto group = std::make_shared<TaskGroup>(work_size);
    auto shared_function =
        std::make_shared<std::decay_t<Function>>(std::move(function));
    for (size_t i = 0; i < work_size; ++i) {
      enqueue([group, shared_function, i]() {
        std::exception_ptr error;
        try {
          (*shared_function)(i);
        } catch (...) {
          error = std::current_exception();
        }
        group->complete(error);
      }, priority);
    }
    group->wait();
  }

  class TaskGroup {
  public:
    explicit TaskGroup(size_t task_count) : remaining_(task_count) {}

    void complete(std::exception_ptr error) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (error && !error_)
        error_ = error;
      if (--remaining_ == 0)
        condition_.notify_one();
    }

    void wait() {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this]() { return remaining_ == 0; });
      if (error_)
        std::rethrow_exception(error_);
    }

  private:
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t remaining_;
    std::exception_ptr error_;
  };

  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        task_condition_.wait(
            lock, [this]() { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty())
          return;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  size_t thread_count_;
  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable task_condition_;
  bool stopping_ = false;
};

} // namespace neural_acd
