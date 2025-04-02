#ifndef PLS_PLATFORM_H
#define PLS_PLATFORM_H

#include <stdexcept>
#include <string>

/* device Info
Device 0: "NVIDIA GeForce RTX 3060"
  CUDA Driver Version / Runtime Version          12.8 / 11.7
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 12288 MBytes (12884377600 bytes)
  (28) Multiprocessors, (128) CUDA Cores/MP:     3584 CUDA Cores
  GPU Max Clock rate:                            1777 MHz (1.78 GHz)
  Memory Clock rate:                             7501 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 2359296 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               zu bytes
  Total amount of shared memory per block:       zu bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          zu bytes
  Texture alignment:                             zu bytes
  Concurrent copy and kernel execution:          Yes with 5 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.8, CUDA Runtime Version = 11.7, NumDevs = 1, Device0 = NVIDIA GeForce RTX 3060
Result = PASS
*/

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