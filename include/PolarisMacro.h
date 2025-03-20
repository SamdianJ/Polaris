#pragma once

//--------------------------------------------
// 平台检测宏
//--------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
    #define PLS_ENGINE_PLATFORM_WINDOWS 1
    #ifdef _MSC_VER
        #define PLS_ENGINE_COMPILER_MSVC 1
    #endif
#elif defined(__APPLE__)
    #define PLS_ENGINE_PLATFORM_APPLE 1
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE
        #define PLS_ENGINE_PLATFORM_IOS 1
    #else
        #define PLS_ENGINE_PLATFORM_MACOS 1
    #endif
#elif defined(__linux__)
    #define PLS_ENGINE_PLATFORM_LINUX 1
#elif defined(__ANDROID__)
    #define PLS_ENGINE_PLATFORM_ANDROID 1
#else
    #error "Unsupported platform!"
#endif

//--------------------------------------------
// host device 函数
//--------------------------------------------
#ifdef __CUDACC__
#define HOST_DEVICE_FUNC __host__ __device__
#define HOST_FUNC __host__
#else
#define HOST_DEVICE_FUNC
#define HOST_FUNC 
#endif

//--------------------------------------------
// 动态库导入/导出宏（Windows专用）
//--------------------------------------------
#if PLS_ENGINE_PLATFORM_WINDOWS
    #ifdef PLS_ENGINE_DLL_EXPORT
        #define PLS_ENGINE_API __declspec(dllexport)
    #elif defined(PLS_ENGINE_DLL_IMPORT)
        #define PLS_ENGINE_API __declspec(dllimport)
    #else
        #define PLS_ENGINE_API
    #endif
#else
    #define PLS_ENGINE_API
#endif

//--------------------------------------------
// 断言宏（仅在调试模式启用）
//--------------------------------------------
#if defined(PLS_ENGINE_DEBUG)
    #include <cassert>
    #define PLS_ENGINE_ASSERT(expr, msg) assert((expr) && (msg))
    #define PLS_ENGINE_ASSERT_MSG(expr, msg) assert((expr) && (msg))
#else
    #define PLS_ENGINE_ASSERT(expr, msg)
    #define PLS_ENGINE_ASSERT_MSG(expr, msg)
#endif

//--------------------------------------------
// 数学精度控制
//--------------------------------------------
#ifdef PLS_ENGINE_USE_DOUBLE_PRECISION
    using Scalar = double;
    #define PLS_ENGINE_SQRT sqrt
    #define PLS_ENGINE_SIN  sin
    #define PLS_ENGINE_COS  cos
#else
    using Scalar = float;
    #define PLS_ENGINE_SQRT sqrtf
    #define PLS_ENGINE_SIN  sinf
    #define PLS_ENGINE_COS  cosf
#endif

//--------------------------------------------
// 内存对齐（SIMD优化）
//--------------------------------------------
#if defined(_MSC_VER)
    #define PLS_ENGINE_ALIGN(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
    #define PLS_ENGINE_ALIGN(n) __attribute__((aligned(n)))
#else
    #define PLS_ENGINE_ALIGN(n)
#endif

// 默认16字节对齐（适合SSE）
#define PLS_ENGINE_ALIGN16 PLS_ENGINE_ALIGN(16)

//--------------------------------------------
// 内联优化
//--------------------------------------------
#if defined(_MSC_VER)
    #define PLS_ENGINE_INLINE __forceinline
    #define PLS_ENGINE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    #define PLS_ENGINE_INLINE inline __attribute__((always_inline))
    #define PLS_ENGINE_NOINLINE __attribute__((noinline))
#else
    #define PLS_ENGINE_INLINE inline
    #define PLS_ENGINE_NOINLINE
#endif

//--------------------------------------------
// 禁用编译器警告（局部使用）
//--------------------------------------------
#if defined(_MSC_VER)
    #define PLS_ENGINE_DISABLE_WARNING(warningNumber) \
        __pragma(warning(push)) \
        __pragma(warning(disable: warningNumber))
    #define PLS_ENGINE_RESTORE_WARNING __pragma(warning(pop))
#elif defined(__GNUC__) || defined(__clang__)
    #define PLS_ENGINE_DISABLE_WARNING(warningName) \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"#warningName\"")
    #define PLS_ENGINE_RESTORE_WARNING _Pragma("GCC diagnostic pop")
#else
    #define PLS_ENGINE_DISABLE_WARNING(warning)
    #define PLS_ENGINE_RESTORE_WARNING
#endif

//--------------------------------------------
// 标记函数/类已弃用
//--------------------------------------------
#if defined(_MSC_VER)
    #define PLS_ENGINE_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__) || defined(__clang__)
    #define PLS_ENGINE_DEPRECATED __attribute__((deprecated))
#else
    #define PLS_ENGINE_DEPRECATED
#endif

#ifdef PLS_ENGINE_COMPILER_MSVC
    #define PTRRESTRICT __restrict
#else
    #define PTRRESTRICT __restrict__
#endif


#define PLS_FOR_I(n)			for (std::size_t i = 0; i < n; ++i)
#define PLS_FOR_I64(n)		for (std::uint64_t i = 0; i < n; ++i)
#define PLS_FOR_J(n)			for (std::size_t j = 0; j < n; ++j)
#define PLS_FOR_K(n)			for (std::size_t k = 0; k < n; ++k)
#define PLS_FOR_RANGE_I(s,n)	for (std::int32_t i = s; i < n; ++i)
#define PLS_FOR_RANGE_J(s,n)	for (std::int32_t j = s; j < n; ++j)

