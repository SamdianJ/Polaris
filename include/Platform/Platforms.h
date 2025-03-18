#ifndef PLS_PLATFORM_H
#define PLS_PLATFORM_H

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                "CUDA error: " + std::string(cudaGetErrorName(err)) + " - " + \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

namespace Polaris
{
	enum Platform
	{
		CPU,
		CUDA
	};
}

#endif