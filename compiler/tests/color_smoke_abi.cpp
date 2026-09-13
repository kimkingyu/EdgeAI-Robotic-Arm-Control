// Test-only M1 ABI. Pixel computation lives in color_smoke.mlir, not this file.
#include <cstdint>
#include <limits>

namespace {
struct MemRef3 {
  std::uint8_t* allocated;
  std::uint8_t* aligned;
  std::int64_t offset;
  std::int64_t sizes[3];
  std::int64_t strides[3];
};
static_assert(sizeof(void*) == 8, "M1 smoke ABI targets 64-bit hosts");
static_assert(sizeof(MemRef3) == 72, "Unexpected ranked memref descriptor layout");
extern "C" void _mlir_ciface_edge_color_smoke_impl(MemRef3*, MemRef3*);

bool checked_span(std::uint64_t height, std::uint64_t width,
                  std::uint64_t stride, std::uint64_t& span) noexcept {
  const auto row = width * 3;
  if (height > 1 && stride > (std::numeric_limits<std::uint64_t>::max() - row) / (height - 1))
    return false;
  span = (height - 1) * stride + row;
  return true;
}
}

extern "C" int edge_color_smoke_run(
    const std::uint8_t* src, std::uint64_t src_bytes,
    std::uint64_t height, std::uint64_t width, std::uint64_t src_stride,
    std::uint8_t* dst, std::uint64_t dst_bytes, std::uint64_t dst_stride) noexcept {
  if (!src || !dst) return 1;
  if (!height || !width || height > 8192 || width > 8192) return 2;
  const auto max_index = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (src_stride < width * 3 || dst_stride < width * 3 || src_stride > max_index || dst_stride > max_index)
    return 3;
  std::uint64_t src_span = 0, dst_span = 0;
  if (!checked_span(height, width, src_stride, src_span) ||
      !checked_span(height, width, dst_stride, dst_span) ||
      src_span > max_index || dst_span > max_index ||
      src_bytes < src_span || dst_bytes < dst_span) return 4;
  const auto a = reinterpret_cast<std::uintptr_t>(src);
  const auto b = reinterpret_cast<std::uintptr_t>(dst);
  const auto max_address = std::numeric_limits<std::uintptr_t>::max();
  if (a > max_address - src_span || b > max_address - dst_span) return 5;
  if (a < b + dst_span && b < a + src_span) return 5;
  auto* mutable_src = const_cast<std::uint8_t*>(src);  // The MLIR kernel only loads it.
  MemRef3 input{mutable_src, mutable_src, 0,
               {static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3},
               {static_cast<std::int64_t>(src_stride), 3, 1}};
  MemRef3 output{dst, dst, 0,
                {static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3},
                {static_cast<std::int64_t>(dst_stride), 3, 1}};
  _mlir_ciface_edge_color_smoke_impl(&input, &output);
  return 0;
}

#ifdef EDGEAI_COLOR_SMOKE_MAIN
#include <algorithm>
#include <iostream>
#include <vector>
int main() {
  constexpr std::uint64_t height = 4, width = 7, in_stride = 26, out_stride = 24;
  std::vector<std::uint8_t> src(height * in_stride, 0xA5), dst(height * out_stride + 16, 0xCC);
  for (std::uint64_t y = 0; y < height; ++y)
    for (std::uint64_t x = 0; x < width; ++x)
      for (std::uint64_t c = 0; c < 3; ++c)
        src[y * in_stride + x * 3 + c] = static_cast<std::uint8_t>(y * 39 + x * 7 + c * 61);
  const auto before = src;
  if (edge_color_smoke_run(src.data(), src.size(), height, width, in_stride,
                          dst.data() + 8, height * out_stride, out_stride) != 0) return 1;
  for (std::uint64_t y = 0; y < height; ++y) {
    for (std::uint64_t x = 0; x < width; ++x)
      for (std::uint64_t c = 0; c < 3; ++c)
        if (dst[8 + y * out_stride + x * 3 + c] != src[y * in_stride + x * 3 + 2 - c]) return 2;
    for (std::uint64_t x = width * 3; x < out_stride; ++x)
      if (dst[8 + y * out_stride + x] != 0xCC) return 3;
  }
  if (src != before || !std::all_of(dst.begin(), dst.begin() + 8, [](auto v) { return v == 0xCC; }) ||
      !std::all_of(dst.end() - 8, dst.end(), [](auto v) { return v == 0xCC; })) return 4;
  const auto valid_output = dst;
  if (edge_color_smoke_run(src.data(), src.size(), height, width, in_stride,
                          dst.data() + 8, 1, out_stride) != 4 || dst != valid_output) return 5;
  std::cout << "C++ MLIR color smoke: exact pixels, input unchanged, padding/guards intact, short buffer rejected\n";
  return 0;
}
#endif
