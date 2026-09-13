#ifndef EDGEAI_PREPROCESS_H
#define EDGEAI_PREPROCESS_H
#include <stdint.h>

#if defined(_WIN32)
#define EDGEAI_API __declspec(dllexport)
#else
#define EDGEAI_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
#define EDGEAI_NOEXCEPT noexcept
extern "C" {
#else
#define EDGEAI_NOEXCEPT
#endif

typedef struct EdgePreprocessContext EdgePreprocessContext;
typedef struct EdgePreprocessSpec {
  uint32_t abi_version;
  uint32_t reserved;
  uint64_t input_height, input_width, output_height, output_width;
} EdgePreprocessSpec;
typedef struct EdgePreprocessRunStats {
  uint64_t allocation_calls, free_calls, requested_bytes, outstanding_allocations;
} EdgePreprocessRunStats;

enum EdgePreprocessStatus {
  EDGEAI_OK = 0, EDGEAI_INVALID_ARGUMENT = 1, EDGEAI_INVALID_DIMENSIONS = 2,
  EDGEAI_INVALID_STRIDE = 3, EDGEAI_INVALID_CAPACITY = 4, EDGEAI_OVERLAP = 5,
  EDGEAI_CONTEXT_ALLOCATION_FAILED = 6, EDGEAI_AUDIT_FAILED = 7
};

EDGEAI_API uint32_t edge_preprocess_abi_version(void) EDGEAI_NOEXCEPT;
EDGEAI_API const char* edge_preprocess_semantic_version(void) EDGEAI_NOEXCEPT;
EDGEAI_API const char* edge_preprocess_variant(void) EDGEAI_NOEXCEPT;
EDGEAI_API uint32_t edge_preprocess_allocation_audit_enabled(void) EDGEAI_NOEXCEPT;
EDGEAI_API int edge_preprocess_create(const EdgePreprocessSpec*, EdgePreprocessContext**) EDGEAI_NOEXCEPT;
// Borrowed buffers; exact touched span is (H-1)*row_stride + W*3. No hidden copy.
// One context is sequential-use only; distinct contexts can run on distinct threads.
// The caller owns buffers/context and must keep them alive until synchronous return.
EDGEAI_API int edge_preprocess_run(EdgePreprocessContext*, const uint8_t* src,
    uint64_t src_bytes, uint64_t src_row_stride, uint8_t* dst,
    uint64_t dst_bytes, uint64_t dst_row_stride) EDGEAI_NOEXCEPT;
EDGEAI_API int edge_preprocess_last_run_stats(const EdgePreprocessContext*,
    EdgePreprocessRunStats*) EDGEAI_NOEXCEPT;
EDGEAI_API void edge_preprocess_destroy(EdgePreprocessContext*) EDGEAI_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#endif
