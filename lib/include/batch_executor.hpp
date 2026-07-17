#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace neural_acd {

class BatchExecutor {
public:
  explicit BatchExecutor(size_t thread_count)
      : thread_count_(std::max<size_t>(1, thread_count)) {
    initialize_worker_placements();
    queues_.reserve(thread_count_);
    for (size_t i = 0; i < thread_count_; ++i)
      queues_.push_back(std::make_unique<WorkerQueue>());
    workers_.reserve(thread_count_);
    for (size_t i = 0; i < thread_count_; ++i)
      workers_.emplace_back([this, i]() { worker_loop(i); });
  }

  ~BatchExecutor() {
    stopping_.store(true, std::memory_order_release);
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
  struct WorkerPlacement {
    size_t node = 0;
    int cpu = -1;
  };

  struct WorkerQueue {
    std::deque<std::function<void()>> tasks;
    std::mutex mutex;
  };

#ifdef __linux__
  static std::vector<int> parse_cpu_list(const std::string &text) {
    std::vector<int> cpus;
    std::istringstream input(text);
    std::string range;
    while (std::getline(input, range, ',')) {
      if (range.empty())
        continue;
      const size_t separator = range.find('-');
      try {
        const int first = std::stoi(range.substr(0, separator));
        const int last = separator == std::string::npos
                             ? first
                             : std::stoi(range.substr(separator + 1));
        for (int cpu = first; cpu <= last; ++cpu)
          cpus.push_back(cpu);
      } catch (const std::exception &) {
        return {};
      }
    }
    return cpus;
  }
#endif

  void initialize_worker_placements() {
    std::vector<std::vector<int>> nodes;
#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool have_affinity =
        sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    size_t missing_after_last_node = 0;
    bool found_node = false;
    for (size_t node = 0; node < 1024; ++node) {
      std::ifstream input("/sys/devices/system/node/node" +
                          std::to_string(node) + "/cpulist");
      if (!input) {
        if (found_node && ++missing_after_last_node >= 32)
          break;
        continue;
      }
      found_node = true;
      missing_after_last_node = 0;
      std::string text;
      std::getline(input, text);
      std::vector<int> cpus = parse_cpu_list(text);
      if (have_affinity) {
        cpus.erase(std::remove_if(cpus.begin(), cpus.end(), [&](int cpu) {
                     return cpu < 0 || cpu >= CPU_SETSIZE ||
                            !CPU_ISSET(cpu, &allowed);
                   }),
                   cpus.end());
      }
      if (!cpus.empty())
        nodes.push_back(std::move(cpus));
    }
    if (nodes.empty() && have_affinity) {
      std::vector<int> cpus;
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed))
          cpus.push_back(cpu);
      }
      if (!cpus.empty())
        nodes.push_back(std::move(cpus));
    }
#endif
    if (nodes.empty())
      nodes.push_back({});

    placements_.resize(thread_count_);
    for (size_t worker = 0; worker < thread_count_; ++worker) {
      const size_t node = worker % nodes.size();
      placements_[worker].node = node;
      if (!nodes[node].empty()) {
        const size_t local_worker = worker / nodes.size();
        placements_[worker].cpu =
            nodes[node][local_worker % nodes[node].size()];
      }
    }
    use_work_stealing_ = nodes.size() > 1;
    const char *mode = std::getenv("VISACD_WORK_STEALING");
    if (mode && std::string(mode) == "1")
      use_work_stealing_ = true;
    else if (mode && std::string(mode) == "0")
      use_work_stealing_ = false;
    bind_workers_ = use_work_stealing_ && nodes.size() > 1;
  }

  void bind_worker(size_t worker) const {
#ifdef __linux__
    if (!bind_workers_ || placements_[worker].cpu < 0)
      return;
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(placements_[worker].cpu, &affinity);
    // Affinity is an optimization only. Restricted containers may reject it.
    pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
#else
    (void)worker;
#endif
  }

  void enqueue(std::function<void()> task, bool priority) {
    if (stopping_.load(std::memory_order_acquire))
      throw std::runtime_error("Cannot submit work to a stopped executor");
    size_t worker = thread_count_;
    if (use_work_stealing_) {
      if (current_executor_ == this) {
        worker = current_worker_;
      } else {
        worker = next_queue_.fetch_add(1, std::memory_order_relaxed) %
                 thread_count_;
      }
    }
    WorkerQueue &queue = worker == thread_count_
                             ? central_queue_
                             : *queues_[worker];
    {
      std::lock_guard<std::mutex> lock(queue.mutex);
      if (stopping_.load(std::memory_order_acquire))
        throw std::runtime_error("Cannot submit work to a stopped executor");
      if (priority)
        queue.tasks.push_front(std::move(task));
      else
        queue.tasks.push_back(std::move(task));
      pending_tasks_.fetch_add(1, std::memory_order_release);
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

  bool take_from_queue(size_t worker, std::function<void()> &task,
                       bool steal = false) {
    WorkerQueue &queue = *queues_[worker];
    std::unique_lock<std::mutex> lock(queue.mutex, std::defer_lock);
    if (steal) {
      if (!lock.try_lock())
        return false;
    } else {
      lock.lock();
    }
    if (queue.tasks.empty())
      return false;
    if (steal) {
      task = std::move(queue.tasks.back());
      queue.tasks.pop_back();
    } else {
      task = std::move(queue.tasks.front());
      queue.tasks.pop_front();
    }
    return true;
  }

  bool take_task(size_t worker, std::function<void()> &task) {
    if (!use_work_stealing_) {
      std::lock_guard<std::mutex> lock(central_queue_.mutex);
      if (central_queue_.tasks.empty())
        return false;
      task = std::move(central_queue_.tasks.front());
      central_queue_.tasks.pop_front();
      return true;
    }
    if (take_from_queue(worker, task))
      return true;

    const size_t local_node = placements_[worker].node;
    for (size_t distance = 1; distance < thread_count_; ++distance) {
      const size_t candidate = (worker + distance) % thread_count_;
      if (placements_[candidate].node == local_node &&
          take_from_queue(candidate, task, true)) {
        return true;
      }
    }
    for (size_t distance = 1; distance < thread_count_; ++distance) {
      const size_t candidate = (worker + distance) % thread_count_;
      if (placements_[candidate].node != local_node &&
          take_from_queue(candidate, task, true)) {
        return true;
      }
    }
    return false;
  }

  void worker_loop(size_t worker) {
    current_executor_ = this;
    current_worker_ = worker;
    bind_worker(worker);
    while (true) {
      std::function<void()> task;
      if (take_task(worker, task)) {
        pending_tasks_.fetch_sub(1, std::memory_order_acq_rel);
        task();
        continue;
      }
      std::unique_lock<std::mutex> lock(condition_mutex_);
      task_condition_.wait(lock, [this]() {
        return stopping_.load(std::memory_order_acquire) ||
               pending_tasks_.load(std::memory_order_acquire) != 0;
      });
      if (stopping_.load(std::memory_order_acquire) &&
          pending_tasks_.load(std::memory_order_acquire) == 0) {
        current_executor_ = nullptr;
        return;
      }
    }
  }

  size_t thread_count_;
  std::vector<std::thread> workers_;
  WorkerQueue central_queue_;
  std::vector<std::unique_ptr<WorkerQueue>> queues_;
  std::vector<WorkerPlacement> placements_;
  std::atomic<size_t> next_queue_{0};
  std::atomic<size_t> pending_tasks_{0};
  std::atomic<bool> stopping_{false};
  std::mutex condition_mutex_;
  std::condition_variable task_condition_;
  bool bind_workers_ = false;
  bool use_work_stealing_ = false;

  inline static thread_local BatchExecutor *current_executor_ = nullptr;
  inline static thread_local size_t current_worker_ = 0;
};

} // namespace neural_acd
