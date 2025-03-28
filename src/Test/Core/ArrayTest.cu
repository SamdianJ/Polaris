#include "Test/Core/ArrayTest.h"
#include <cassert>
#include "Platform/Platforms.h"

namespace Polaris
{
    namespace Device
    {
#ifdef __CUDACC__
        // 一个简单的 CUDA kernel，用于将数组中所有元素增加一个给定的值
        KERNEL_FUNC void KernelIncrement(Polaris::Array<int, 10>* arr, int inc) {
            // 本测试使用单线程处理，遍历整个数组
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                for (size_t i = 0; i < arr->size(); ++i) {
                    (*arr)[i] += inc;
                }
            }
        }
#endif
    }
}

void Polaris::Device::TestArray() {
#ifdef __CUDACC__

    PLS_INFO("==== GPU Array Test ====");
    // 使用 CUDA Unified Memory 分配 GPU 上的 Array 对象
    Array<int, 10>* gpuArray = nullptr;
    CUDA_CHECK(cudaMallocManaged(&gpuArray, sizeof(Array<int, 10>)));

    // 使用 placement new 构造 Array 对象
    new (gpuArray) Array<int, 10>();
    // 填充数组，所有元素设为 10
    gpuArray->fill(10);

    // 启动 CUDA kernel，将数组中所有元素增加 3
    KernelIncrement << <1, 1 >> > (gpuArray, 3);
    cudaDeviceSynchronize();

    // 验证数据，期望每个元素值为 13
    for (size_t i = 0; i < gpuArray->size(); ++i) {
        assert((*gpuArray)[i] == 13);
    }

    PLS_INFO("GPU Array Test passed");

    // 调用析构函数（如果有必要）并释放 Unified Memory
    gpuArray->~Array<int, 10>();
    cudaFree(gpuArray);
#endif
}