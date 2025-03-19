#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include "../thirdparty/fmt/include/fmt/format.h"  // 需要安装 fmt 库
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// 初始化 Windows 控制台以支持 ANSI 转义码
inline void LogInit() {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE)
        return;
    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode))
        return;
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
#endif
}

// 获取当前时间字符串（包含毫秒）
inline std::string CurrentTimeString() {
    using namespace std::chrono;

    // 获取当前时间点
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);

    // 转换成本地时间
    struct tm local_time;
#ifdef _WIN32
    localtime_s(&local_time, &now_time);
#else
    localtime_r(&now_time, &local_time);
#endif

    // 获取毫秒部分
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // 格式化时间字符串
    return fmt::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        local_time.tm_year + 1900,  // 年份需要加1900
        local_time.tm_mon + 1,      // 月份从0开始
        local_time.tm_mday,
        local_time.tm_hour,
        local_time.tm_min,
        local_time.tm_sec,
        ms.count());
}

// ANSI 颜色码定义
#define LOG_COLOR_RESET   "\033[0m"
#define LOG_COLOR_RED     "\033[31m"
#define LOG_COLOR_GREEN   "\033[32m"
#define LOG_COLOR_YELLOW  "\033[33m"
#define LOG_COLOR_CYAN    "\033[36m"

// 格式化日志宏：添加时间戳
#define PLS_INFO(fmt_str, ...)  do { \
    std::cout << LOG_COLOR_GREEN << "[" << CurrentTimeString() << "] POLARIS_EIGINE [INFO] " \
              << fmt::format(fmt_str, ##__VA_ARGS__) \
              << LOG_COLOR_RESET << std::endl; \
} while(0)

#define PLS_WARN(fmt_str, ...)  do { \
    std::cout << LOG_COLOR_YELLOW << "[" << CurrentTimeString() << "] POLARIS_EIGINE [WARN] " \
              << fmt::format(fmt_str, ##__VA_ARGS__) \
              << LOG_COLOR_RESET << std::endl; \
} while(0)

#define PLS_ERROR(fmt_str, ...) do { \
    std::cerr << LOG_COLOR_RED << "[" << CurrentTimeString() << "] POLARIS_EIGINE [ERROR] " \
              << fmt::format(fmt_str, ##__VA_ARGS__) \
              << LOG_COLOR_RESET << std::endl; \
} while(0)

#define PLS_DEBUG(fmt_str, ...) do { \
    std::cout << LOG_COLOR_CYAN << "[" << CurrentTimeString() << "] POLARIS_EIGINE [DEBUG] " \
              << fmt::format(fmt_str, ##__VA_ARGS__) \
              << LOG_COLOR_RESET << std::endl; \
} while(0)