#include "Core/LinearAlgebra/gemm.cuh"
#include "Platform/CUDA/kernelLaunch.h"

KERNEL_FUNC void gemm_naive(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	size_t x_id = blockIdx.y * blockDim.y + threadIdx.y;
	size_t y_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (x_id >= m || y_id >= n)
		return;

	Scalar tmp = Scalar(0);

	PLS_PRAGMA_UNROLL
	PLS_FOR_I(k)
	{
		tmp += a[x_id * k + i] * b[i * n + y_id];
	}

	c[x_id * n + y_id] = tmp;
}

HOST_FUNC cudaError_t Polaris::gemm_kernel(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	constexpr Label dimx = 16;
	constexpr Label dimy = 16;
	dim3 dimBlock(dimx, dimy);
	dim3 dimGrid(
		(n + dimBlock.x - 1) / dimBlock.x, 
		(m + dimBlock.y - 1) / dimBlock.y);
	gemm_naive << < dimGrid, dimBlock >> > (m, k, n, a, b, c);
	cudaError_t launchError = cudaPeekAtLastError();
	cudaError_t syncError = cudaDeviceSynchronize();
	return (launchError != cudaSuccess) ? launchError : syncError;
}
