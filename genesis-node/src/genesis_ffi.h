#ifndef GENESIS_FFI_H
#define GENESIS_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RuntimeOpaque RuntimeOpaque;

// Opaque handle for the Rust Runtime
typedef struct {
    RuntimeOpaque* ptr;
} RuntimeHandle;

// FFI API for Runtime control
RuntimeHandle genesis_runtime_load(const char* path, const char* backend);
void genesis_runtime_free(RuntimeHandle handle);
void genesis_runtime_tick(RuntimeHandle handle, const int32_t* inputs, bool* output_spikes);
void genesis_runtime_inject_text(RuntimeHandle handle, const char* text);
uint32_t genesis_runtime_neuron_count(RuntimeHandle handle);

#ifdef __cplusplus
}
#endif

#endif // GENESIS_FFI_H
