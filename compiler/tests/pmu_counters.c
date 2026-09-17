// 用 perf_event_open 直接读 ARMv8 PMU 计数器，不依赖 perf 工具、不需要 root。
// paranoid=2 下允许测量自己的进程，这正是我们需要的范围。
//
// 目的：判断 MLIR 预处理慢于 OpenCV 的主要来源是访存还是指令执行。
// 只报硬件实测计数，不做任何外推。
#define _GNU_SOURCE
#include <asm/unistd.h>
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef struct {
  const char *name;
  uint32_t type;
  uint64_t config;
  int fd;
  uint64_t value;
  int supported;
} Counter;

// PERF_COUNT_HW_* 是通用事件，由内核映射到 ARMv8 具体事件号。
static Counter counters[] = {
    {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, 0, 0},
    {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1, 0, 0},
    {"cache_references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, -1, 0, 0},
    {"cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, -1, 0, 0},
    {"branch_instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, -1, 0, 0},
    {"branch_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, -1, 0, 0},
    {"stalled_cycles_backend", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND, -1, 0, 0},
    {"stalled_cycles_frontend", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, -1, 0, 0},
    {"l1d_read_misses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
         (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), -1, 0, 0},
    {"l1d_read_accesses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
         (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), -1, 0, 0},
    {"ll_read_misses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
         (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), -1, 0, 0},
    {"ll_read_accesses", PERF_TYPE_HW_CACHE,
     PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
         (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), -1, 0, 0},
    // 通用 BRANCH_INSTRUCTIONS 在本内核上不可用，改用 ARMv8 原生事件号：
    // 0x0D BR_IMMED_RETIRED（立即分支）、0x10 BR_MIS_PRED（分支误测）。
    // 用原始事件号而非伪造缺失值，这样分支能否排除是有据可查的。
    {"armv8_br_immed_retired", PERF_TYPE_RAW, 0x0D, -1, 0, 0},
    {"armv8_br_mis_pred", PERF_TYPE_RAW, 0x10, -1, 0, 0},
    // 0x14 L1I_CACHE，用来确认取指侧是否为隐藏瓶颈（指令数暴增时值得一看）。
    {"armv8_l1i_cache", PERF_TYPE_RAW, 0x14, -1, 0, 0},
    {"armv8_l1i_cache_refill", PERF_TYPE_RAW, 0x01, -1, 0, 0},
};

static const int counter_count = sizeof(counters) / sizeof(counters[0]);

// 只打开 [begin, end) 区间的事件。ARMv8 PMU 通用计数器只有 6 个，一次全开会
// 触发复用，缩放后精度下降、部分事件甚至读不到。分组测量换取可信计数。
static int open_begin = 0, open_end = 0;

static long perf_open(struct perf_event_attr *attr, pid_t pid, int cpu, int group) {
  return syscall(__NR_perf_event_open, attr, pid, cpu, group, 0);
}

int pmu_open_range(int begin, int end) {
  if (begin < 0) begin = 0;
  if (end > counter_count) end = counter_count;
  open_begin = begin;
  open_end = end;
  int opened = 0;
  for (int i = 0; i < counter_count; ++i) {
    counters[i].fd = -1;
    counters[i].supported = 0;
  }
  for (int i = begin; i < end; ++i) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.type = counters[i].type;
    attr.config = counters[i].config;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    counters[i].fd = (int)perf_open(&attr, 0, -1, -1);
    // 不支持的事件记为 unsupported，而不是伪造一个 0 值。
    counters[i].supported = counters[i].fd >= 0;
    if (counters[i].supported) ++opened;
  }
  return opened;
}

int pmu_open(void) { return pmu_open_range(0, counter_count); }

void pmu_reset_and_enable(void) {
  for (int i = 0; i < counter_count; ++i) {
    if (!counters[i].supported) continue;
    ioctl(counters[i].fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(counters[i].fd, PERF_EVENT_IOC_ENABLE, 0);
  }
}

void pmu_disable(void) {
  for (int i = 0; i < counter_count; ++i)
    if (counters[i].supported) ioctl(counters[i].fd, PERF_EVENT_IOC_DISABLE, 0);
}

// 返回 0 成功。values 需能容纳 counter_count 个元素；不支持的事件写 UINT64_MAX。
int pmu_read(uint64_t *values, double *scaling_min) {
  *scaling_min = 1.0;
  for (int i = 0; i < counter_count; ++i) {
    if (!counters[i].supported) {
      values[i] = UINT64_MAX;
      continue;
    }
    uint64_t buffer[3] = {0, 0, 0};
    ssize_t got = read(counters[i].fd, buffer, sizeof(buffer));
    if (got < (ssize_t)sizeof(buffer)) return -1;
    uint64_t raw = buffer[0], enabled = buffer[1], running = buffer[2];
    if (running == 0) {
      values[i] = UINT64_MAX;
      continue;
    }
    // 计数器被复用时按实际运行时间缩放；缩放比例一并上报，便于判断可信度。
    double ratio = (double)running / (double)enabled;
    if (ratio < *scaling_min) *scaling_min = ratio;
    values[i] = (uint64_t)((double)raw * (double)enabled / (double)running);
  }
  return 0;
}

int pmu_count(void) { return counter_count; }
const char *pmu_name(int index) {
  return (index >= 0 && index < counter_count) ? counters[index].name : "";
}
int pmu_supported(int index) {
  return (index >= 0 && index < counter_count) ? counters[index].supported : 0;
}

void pmu_close(void) {
  for (int i = 0; i < counter_count; ++i) {
    if (counters[i].fd >= 0) close(counters[i].fd);
    counters[i].fd = -1;
    counters[i].supported = 0;
  }
}
