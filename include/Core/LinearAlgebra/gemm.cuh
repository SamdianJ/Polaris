#ifndef PHYSICSENGINE_SGEMM_CUH
#define PHYSICSENGINE_SGEMM_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Polaris.h"

namespace Polaris
{
	HOST_FUNC cudaError_t gemm_naive_kernel(size_t m, size_t n, size_t k, const Scalar* a, const Scalar* b, Scalar* c);
	HOST_FUNC cudaError_t gemm_block_transpose_kernel(size_t m, size_t n, size_t k, const Scalar* a, const Scalar* b, Scalar* c);
	HOST_FUNC cudaError_t gemm_smem_kernel(size_t m, size_t n, size_t k, const Scalar* a, const Scalar* b, Scalar* c);
	HOST_FUNC cudaError_t gemm_smem_tiled_kernel(size_t m, size_t n, size_t k, const Scalar* a, const Scalar* b, Scalar* c);
	HOST_FUNC cudaError_t gemm_smem_tiled_1D_kernel(size_t m, size_t n, size_t k, const Scalar* a, const Scalar* b, Scalar* c);
}

#endif