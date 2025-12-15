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


// State destruction function type: iterate all SIMD blocks and call destructor for each state object
// states: pointer to state memory
// num_blocks: number of SIMD blocks
// block_size: state size per SIMD block
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
    size_t state_size;  // State size per SIMD block (stream mode)
    DestroyStatesFn destroy_states;  // State destruction function (stream mode)
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