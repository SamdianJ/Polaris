#include "Test/Core/AllocatorTest.h"
#include <iostream>
#include <vector>
#include <cassert>
#include "Platform/Platforms.h"

using namespace Polaris;

PlsInt32 Device::Test_Allocator()
{
    const size_t N = 10;
    cuda_utils::CudaAllocator<int> allocator;

    int* device_ptr = allocator.allocate(N);

    std::vector<int> host_data(N);
    for (size_t i = 0; i < N; ++i) {
        host_data[i] = static_cast<int>(i * 3);
    }

    cudaError_t err = cudaMemcpy(device_ptr, host_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy to device failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::vector<int> result_data(N, 0);
    err = cudaMemcpy(result_data.data(), device_ptr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy from device failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    for (size_t i = 0; i < N; ++i) {
        assert(result_data[i] == static_cast<int>(i * 3));
    }
    std::cout << "CudaAllocator test passed!" << std::endl;

    allocator.deallocate(device_ptr, N);
    return 0;
}


