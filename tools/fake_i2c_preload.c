/*
 * 假 I2C 总线（LD_PRELOAD 注入），用于在无硬件时实跑主程序的完整下发路径。
 *
 * 为什么需要：主程序的 setJointAngles 下发分支此前从未被执行过 ——
 * 单元测试验证的是驱动类本身，而"主程序在舵机连通时究竟写了什么寄存器、
 * 对应什么角度"是另一回事。等接线后才第一次跑这条路径，任何错误都会
 * 直接作用在实物舵机上。
 *
 * 本文件只拦截对指定假设备路径的调用，其余 fd 一律透传给真实系统调用，
 * 因此不会影响主程序读模型、写日志等正常 IO。
 *
 * 构建：
 *   gcc -shared -fPIC -o fake_i2c.so tools/fake_i2c_preload.c -ldl
 * 使用：
 *   FAKE_I2C_PATH=/tmp/fake-i2c FAKE_I2C_LOG=/tmp/i2c_writes.log \
 *   LD_PRELOAD=./fake_i2c.so ./bin/edge_arm_control --i2c /tmp/fake-i2c --drive
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define FAKE_FD_BASE 0x7A00

static int (*real_open)(const char *, int, ...) = NULL;
static ssize_t (*real_write)(int, const void *, size_t) = NULL;
static ssize_t (*real_read)(int, void *, size_t) = NULL;
static int (*real_close)(int) = NULL;
static int (*real_ioctl)(int, unsigned long, ...) = NULL;

static uint8_t g_regs[256];
static uint8_t g_cursor = 0;
static int g_fake_fd = -1;
static FILE *g_log = NULL;

static void init_syms(void) {
    if (real_open) return;
    real_open = dlsym(RTLD_NEXT, "open");
    real_write = dlsym(RTLD_NEXT, "write");
    real_read = dlsym(RTLD_NEXT, "read");
    real_close = dlsym(RTLD_NEXT, "close");
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    /* 上电默认值，与 NXP 数据手册一致 */
    g_regs[0x00] = 0x11;
    g_regs[0xFE] = 0x1E;
    const char *log_path = getenv("FAKE_I2C_LOG");
    if (log_path) g_log = fopen(log_path, "w");
}

static int is_fake_path(const char *path) {
    const char *want = getenv("FAKE_I2C_PATH");
    return want && path && strcmp(path, want) == 0;
}

int open(const char *path, int flags, ...) {
    init_syms();
    if (is_fake_path(path)) {
        g_fake_fd = FAKE_FD_BASE;
        if (g_log) fprintf(g_log, "OPEN %s\n", path);
        return g_fake_fd;
    }
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    return real_open(path, flags, mode);
}

int ioctl(int fd, unsigned long request, ...) {
    init_syms();
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    if (fd == g_fake_fd && g_fake_fd >= 0) {
        if (g_log) fprintf(g_log, "IOCTL 0x%lx\n", request);
        return 0;
    }
    return real_ioctl(fd, request, arg);
}

ssize_t write(int fd, const void *buf, size_t count) {
    init_syms();
    if (fd != g_fake_fd || g_fake_fd < 0) return real_write(fd, buf, count);
    if (count == 0) return -1;

    const uint8_t *bytes = (const uint8_t *)buf;
    g_cursor = bytes[0];
    for (size_t i = 1; i < count; ++i) g_regs[(uint8_t)(g_cursor + i - 1)] = bytes[i];

    if (g_log) {
        fprintf(g_log, "WRITE reg=0x%02X len=%zu data=", bytes[0], count - 1);
        for (size_t i = 1; i < count; ++i) fprintf(g_log, "%02X ", bytes[i]);
        fprintf(g_log, "\n");
        fflush(g_log);
    }
    return (ssize_t)count;
}

ssize_t read(int fd, void *buf, size_t count) {
    init_syms();
    if (fd != g_fake_fd || g_fake_fd < 0) return real_read(fd, buf, count);
    if (count != 1) return -1;
    *(uint8_t *)buf = g_regs[g_cursor];
    return 1;
}

int close(int fd) {
    init_syms();
    if (fd == g_fake_fd && g_fake_fd >= 0) {
        if (g_log) { fprintf(g_log, "CLOSE\n"); fflush(g_log); }
        g_fake_fd = -1;
        return 0;
    }
    return real_close(fd);
}
