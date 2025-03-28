#include "Test/Core/ArrayTest.h"
#include <cassert>
#include "Platform/Platforms.h"

namespace Polaris
{
    namespace Device
    {
#ifdef __CUDACC__
        // һ���򵥵� CUDA kernel�����ڽ�����������Ԫ������һ��������ֵ
        KERNEL_FUNC void KernelIncrement(Polaris::Array<int, 10>* arr, int inc) {
            // ������ʹ�õ��̴߳���������������
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
    // ʹ�� CUDA Unified Memory ���� GPU �ϵ� Array ����
    Array<int, 10>* gpuArray = nullptr;
    CUDA_CHECK(cudaMallocManaged(&gpuArray, sizeof(Array<int, 10>)));

    // ʹ�� placement new ���� Array ����
    new (gpuArray) Array<int, 10>();
    // ������飬����Ԫ����Ϊ 10
    gpuArray->fill(10);

    // ���� CUDA kernel��������������Ԫ������ 3
    KernelIncrement << <1, 1 >> > (gpuArray, 3);
    cudaDeviceSynchronize();

    // ��֤���ݣ�����ÿ��Ԫ��ֵΪ 13
    for (size_t i = 0; i < gpuArray->size(); ++i) {
        assert((*gpuArray)[i] == 13);
    }

    PLS_INFO("GPU Array Test passed");

    // ������������������б�Ҫ�����ͷ� Unified Memory
    gpuArray->~Array<int, 10>();
    cudaFree(gpuArray);
#endif
}