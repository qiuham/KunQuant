#pragma once

#include "Stage.hpp"
#include <memory>
#include <functional>

namespace kun {

enum class MemoryLayout {
    STs,
    TS,
    STREAM,
};


// 状态销毁函数类型：遍历所有 SIMD 块，调用每个状态对象的析构函数
// states: 状态内存起始地址
// num_blocks: SIMD 块数量
// block_size: 每个 SIMD 块的状态大小
using DestroyStatesFn = void (*)(void* states, size_t num_blocks, size_t block_size);

struct Module {
    size_t required_version;
    size_t num_stages;
    Stage *stages;
    size_t num_buffers;
    BufferInfo *buffers;
    MemoryLayout input_layout;
    MemoryLayout output_layout;
    size_t blocking_len;
    Datatype dtype;
    size_t aligned;
    size_t state_size;  // 每个 SIMD 块的状态大小（流式模式）
    DestroyStatesFn destroy_states;  // 状态销毁函数（流式模式）
};

struct Library {
    void *handle;
    std::function<void(Library*)> dtor;
    KUN_API const Module *getModule(const char *name);
    KUN_API static std::shared_ptr<Library> load(const char *filename);
    Library(const Library &) = delete;
    Library(void *handle) : handle{handle} {}
    KUN_API ~Library();
};

} // namespace kun