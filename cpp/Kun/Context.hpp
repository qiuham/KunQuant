#pragma once

#include "Stage.hpp"
#include "StreamBuffer.hpp"
#include <atomic>
#include <memory>
#include <stdlib.h>
#include <stddef.h>
#include <stdexcept>
#include <string>

namespace kun {

struct KUN_API RuntimeStage {
    const Stage *stage;
    Context *ctx;

    std::atomic<size_t> pending;
    std::atomic<size_t> doing_index;
    std::atomic<size_t> done_count;

    RuntimeStage(Stage *stage, Context *ctx) : stage{stage}, ctx{ctx} {
        reset(ctx);
    }

    RuntimeStage(RuntimeStage &&other) noexcept {
        stage = other.stage;
        ctx = other.ctx;
        pending = other.pending.load(std::memory_order_relaxed);
        doing_index = other.doing_index.load(std::memory_order_relaxed);
        done_count = other.done_count.load(std::memory_order_relaxed);
    }

    bool doJob();

    bool hasJobToDo() const {
        auto cur_idx = doing_index.load();
        auto num_tasks = getNumTasks();
        return (cur_idx < num_tasks);
    }
    size_t getNumTasks() const;

    void reset(Context *ctx) {
        pending = stage->orig_pending;
        doing_index = 0;
        done_count = getNumTasks();
    }

    void enqueue();
    // returns true if there may be more tasks in the job
    bool onDone(size_t cnt);
};

struct KUN_API Executor {
    virtual void enqueue(RuntimeStage *stage) = 0;
    virtual void dequeue(RuntimeStage *stage) = 0;
    // virtual bool takeSingleJob() = 0;
    virtual void runUntilDone() = 0;
    virtual ~Executor() = default;
};

#define CHECKED_PTR 0
struct Buffer {
    union {
        void* __restrict raw;
        float *__restrict ptr;
        double *__restrict ptrD;
        StreamBuffer<float> * __restrict stream_buf;
        StreamBuffer<double> * __restrict stream_bufD;
    };
    
    StreamBuffer<float> * getStream(float*) const {
        return stream_buf;
    }
    StreamBuffer<double> * getStream(double*) const {
        return stream_bufD;
    }
    template <typename T>
    T* getPtr() const {
        return reinterpret_cast<T*>(raw);
    }
    size_t num_time; // the dimension in time
#if CHECKED_PTR
    size_t size; // size in bytes
#endif
    std::atomic<int> refcount;

    KUN_API void alloc(size_t count, size_t use_count, size_t elem_size);

    Buffer(size_t num_time) {
        ptr = nullptr;
        this->num_time = num_time;
        refcount = 0;
    }

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&other) noexcept {
        ptr = other.ptr;
        num_time = other.num_time;
        refcount = other.refcount.load();
        other.ptr = nullptr;
    }

    Buffer(float *inptr, size_t num_time) {
        ptr = inptr;
        this->num_time = num_time;
        refcount = -1000;
    }

    void ref() { ++refcount; }

    KUN_API void deref();

    KUN_API ~Buffer();
};

struct Executor;

struct Context {
    std::vector<Buffer> buffers;
    std::vector<RuntimeStage> stages;
    std::shared_ptr<Executor> executor;
    size_t buffer_len;

    size_t stock_count;
    size_t total_time;
    size_t start;
    size_t length;
    size_t simd_len;
    Datatype dtype;
    bool is_stream;

    // 流式模式状态管理
    void* states;
    size_t states_size;
    bool states_initialized;

    // 获取指定 SIMD 块的状态指针
    // offset: 该状态类型的基础偏移
    // stock_idx: SIMD 块索引 (0, 1, 2, ...)
    // block_size: 每个 SIMD 块的总状态大小
    template<typename T>
    T* state_ptr(size_t offset, size_t stock_idx, size_t block_size) {
#ifndef NDEBUG
        // DEBUG 模式边界检查
        if (states == nullptr) {
            throw std::runtime_error("state_ptr: states buffer is null");
        }
        size_t access_end = stock_idx * block_size + offset + sizeof(T);
        if (access_end > states_size) {
            throw std::runtime_error("state_ptr: access out of bounds (offset="
                + std::to_string(offset) + ", stock_idx=" + std::to_string(stock_idx)
                + ", block_size=" + std::to_string(block_size)
                + ", access_end=" + std::to_string(access_end)
                + ", states_size=" + std::to_string(states_size) + ")");
        }
#endif
        return reinterpret_cast<T*>(
            static_cast<char*>(states) + stock_idx * block_size + offset);
    }
};

KUN_API std::shared_ptr<Executor> createSingleThreadExecutor();
KUN_API std::shared_ptr<Executor> createMultiThreadExecutor(int num_threads);

} // namespace kun