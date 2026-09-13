// Thin ABI: all pixel computation is in generated MLIR objects.
#include "edgeai_preprocess.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#ifndef EDGEAI_OUT_HEIGHT
#define EDGEAI_OUT_HEIGHT 640
#endif
#ifndef EDGEAI_OUT_WIDTH
#define EDGEAI_OUT_WIDTH 640
#endif
#ifndef EDGEAI_VARIANT
#define EDGEAI_VARIANT "unknown"
#endif
#ifndef EDGEAI_ALLOCATION_AUDIT
#define EDGEAI_ALLOCATION_AUDIT 0
#endif
static_assert(EDGEAI_OUT_HEIGHT > 0 && EDGEAI_OUT_HEIGHT <= 8192);
static_assert(EDGEAI_OUT_WIDTH > 0 && EDGEAI_OUT_WIDTH <= 8192);
static_assert(sizeof(void*) == 8 && sizeof(std::size_t) == 8);
struct EdgePreprocessContext {
  EdgePreprocessSpec spec;
  EdgePreprocessRunStats stats{};
};
namespace {
struct MemRef3 {
  std::uint8_t* allocated;
  std::uint8_t* aligned;
  std::int64_t offset;
  std::int64_t sizes[3], strides[3];
};
static_assert(sizeof(MemRef3) == 72);
constexpr auto max_index = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
extern "C" void _mlir_ciface_edge_resize_color_impl(MemRef3*, MemRef3*);

bool span(std::uint64_t height, std::uint64_t width, std::uint64_t stride,
          std::uint64_t& bytes) noexcept {
  const auto row = width * 3;
  if (height > 1 && stride > (max_index - row) / (height - 1)) return false;
  bytes = (height - 1) * stride + row;
  return bytes <= max_index;
}
MemRef3 descriptor(std::uint8_t* data, std::uint64_t h, std::uint64_t w,
                   std::uint64_t stride) noexcept {
  return {data, data, 0, {static_cast<std::int64_t>(h), static_cast<std::int64_t>(w), 3},
          {static_cast<std::int64_t>(stride), 3, 1}};
}
#if EDGEAI_ALLOCATION_AUDIT
struct Audit { EdgePreprocessRunStats stats{}; bool error = false; };
thread_local Audit* active_audit = nullptr;
void account_allocation(void* pointer, std::size_t bytes) noexcept {
  if (!active_audit) return;
  if (!pointer) { active_audit->error = true; return; }
  auto& stats = active_audit->stats;
  ++stats.allocation_calls;
  ++stats.outstanding_allocations;
  if (bytes > std::numeric_limits<std::uint64_t>::max() - stats.requested_bytes)
    active_audit->error = true;
  else stats.requested_bytes += bytes;
}
#endif
}
#if EDGEAI_ALLOCATION_AUDIT
extern "C" void* __real_malloc(std::size_t);
extern "C" void* __real_aligned_alloc(std::size_t, std::size_t);
extern "C" void __real_free(void*);
extern "C" void* __wrap_malloc(std::size_t bytes) {
  void* result = __real_malloc(bytes);
  account_allocation(result, bytes);
  return result;
}
extern "C" void* __wrap_aligned_alloc(std::size_t alignment, std::size_t bytes) {
  void* result = __real_aligned_alloc(alignment, bytes);
  account_allocation(result, bytes);
  return result;
}
extern "C" void __wrap_free(void* pointer) {
  if (pointer && active_audit) {
    auto& stats = active_audit->stats;
    ++stats.free_calls;
    if (stats.outstanding_allocations) --stats.outstanding_allocations;
    else active_audit->error = true;
  }
  __real_free(pointer);
}
#endif
extern "C" {
uint32_t edge_preprocess_abi_version() noexcept { return 1; }
const char* edge_preprocess_semantic_version() noexcept { return "linear_half_pixel_q11_v1"; }
const char* edge_preprocess_variant() noexcept { return EDGEAI_VARIANT; }
uint32_t edge_preprocess_allocation_audit_enabled() noexcept { return EDGEAI_ALLOCATION_AUDIT ? 1 : 0; }
int edge_preprocess_create(const EdgePreprocessSpec* spec, EdgePreprocessContext** output) noexcept {
  if (!spec || !output) return EDGEAI_INVALID_ARGUMENT;
  *output = nullptr;
  if (spec->abi_version != 1 || spec->reserved) return EDGEAI_INVALID_ARGUMENT;
  if (!spec->input_height || !spec->input_width || spec->input_height > 8192 || spec->input_width > 8192 ||
      spec->output_height != EDGEAI_OUT_HEIGHT || spec->output_width != EDGEAI_OUT_WIDTH)
    return EDGEAI_INVALID_DIMENSIONS;
  auto* context = new (std::nothrow) EdgePreprocessContext{*spec, {}};
  if (!context) return EDGEAI_CONTEXT_ALLOCATION_FAILED;
  *output = context;
  return EDGEAI_OK;
}
int edge_preprocess_run(EdgePreprocessContext* context, const uint8_t* src, uint64_t src_bytes,
                       uint64_t src_stride, uint8_t* dst, uint64_t dst_bytes, uint64_t dst_stride) noexcept {
  if (!context || !src || !dst) return EDGEAI_INVALID_ARGUMENT;
  const auto& spec = context->spec;
  if (src_stride < spec.input_width * 3 || src_stride > max_index ||
      dst_stride != spec.output_width * 3) return EDGEAI_INVALID_STRIDE;
  std::uint64_t src_span = 0, dst_span = 0;
  if (!span(spec.input_height, spec.input_width, src_stride, src_span) ||
      !span(spec.output_height, spec.output_width, dst_stride, dst_span) ||
      src_bytes < src_span || dst_bytes < dst_span) return EDGEAI_INVALID_CAPACITY;
  auto a = reinterpret_cast<std::uintptr_t>(src), b = reinterpret_cast<std::uintptr_t>(dst);
  auto address_max = std::numeric_limits<std::uintptr_t>::max();
  if (a > address_max - src_span || b > address_max - dst_span) return EDGEAI_INVALID_CAPACITY;
  if (a < b + dst_span && b < a + src_span) return EDGEAI_OVERLAP;
  auto input = descriptor(const_cast<std::uint8_t*>(src), spec.input_height, spec.input_width, src_stride);
  auto output = descriptor(dst, spec.output_height, spec.output_width, dst_stride);
#if EDGEAI_ALLOCATION_AUDIT
  if (active_audit) return EDGEAI_AUDIT_FAILED;
  Audit audit;
  active_audit = &audit;
#endif
  _mlir_ciface_edge_resize_color_impl(&input, &output);
#if EDGEAI_ALLOCATION_AUDIT
  active_audit = nullptr;
  context->stats = audit.stats;
  if (audit.error || audit.stats.outstanding_allocations) return EDGEAI_AUDIT_FAILED;
#else
  context->stats = {};
#endif
  return EDGEAI_OK;
}
int edge_preprocess_last_run_stats(const EdgePreprocessContext* context, EdgePreprocessRunStats* stats) noexcept {
  if (!context || !stats) return EDGEAI_INVALID_ARGUMENT;
  *stats = context->stats;
  return EDGEAI_OK;
}
void edge_preprocess_destroy(EdgePreprocessContext* context) noexcept { delete context; }
}
