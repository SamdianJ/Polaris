#include "Core/LinearAlgebra/gemm.cuh"
#include "Platform/CUDA/kernelLaunch.h"

KERNEL_FUNC void gemm_naive(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	size_t y_id = blockIdx.x * blockDim.x + threadIdx.x; // transpose block
	size_t x_id = blockIdx.y * blockDim.y + threadIdx.y;

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

template <typename size_t BM, typename size_t BN, typename size_t SW>
KERNEL_FUNC void gemm_smem(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	size_t y_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t x_id = blockIdx.y * blockDim.y + threadIdx.y;

	// deprecated as k can be very large
	//__shared__ Scalar SA[16][k]; // SA[row][col]
	//__shared__ Scalar SB[k][16]; 

	// shared slide
	size_t slide_window = SW;
	size_t num_slide_window = k / slide_window;

	__shared__ Scalar SA[BM][SW]; // SA[row][col]
	__shared__ Scalar SB[SW][BN];

	if (x_id >= m || y_id >= n)
		return;

	Scalar tmp = Scalar(0.0);
	for (size_t slide_id = 0; slide_id < num_slide_window; ++slide_id)
	{
		// load to smem
		if (threadIdx.x < slide_window)
		{
			SA[threadIdx.y][threadIdx.x] = a[x_id  * k + threadIdx.x + slide_id * slide_window];	
		}

		if (threadIdx.y < slide_window)
		{
			SB[threadIdx.y][threadIdx.x] = b[y_id + (threadIdx.y + slide_id * slide_window) * n];
		}
		__syncthreads();

		PLS_PRAGMA_UNROLL
			PLS_FOR_I(slide_window)
		{
			tmp += SA[threadIdx.y][i] * SB[i][threadIdx.x];
		}
		__syncthreads();
	}

	c[x_id * n + y_id] = tmp;
}

template <typename size_t BM, typename size_t BN, typename size_t SW, typename size_t TM, typename size_t TN>
KERNEL_FUNC void gemm_smem_tiled(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	size_t row = TM * (blockIdx.x * blockDim.x + threadIdx.x);
	size_t col = TN * (blockIdx.y * blockDim.y + threadIdx.y);

	Scalar tmp[TM][TN] = { Scalar(0) };
	constexpr size_t slide_window = SW;
	size_t num_slide_window = k / slide_window;

	__shared__ Scalar SA[BM * TM][SW]; // SA[row][col]
	__shared__ Scalar SB[SW][BN * TN];

	if (row >= m || col >= n)
		return;

	for (size_t slide_id = 0; slide_id < num_slide_window; ++slide_id)
	{
		for (size_t thread_local_i = 0; thread_local_i < TM; ++thread_local_i)
		{
			// load to smem
			for (size_t i = threadIdx.y; i < k; i += slide_window)
			{
				size_t local_i = i - slide_id * slide_window;
				if (local_i < slide_window)
				{
					SA[thread_local_i + threadIdx.x * TM][local_i] = a[(row + thread_local_i) * k + i];
				}
			}
		}
		for (size_t thread_local_i = 0; thread_local_i < TN; ++thread_local_i)
		{
			for (size_t i = threadIdx.x; i < k; i += slide_window)
			{
				size_t local_i = i - slide_id * slide_window;
				if (local_i < slide_window)
				{
					SB[local_i][threadIdx.y * TN + thread_local_i] = b[col + thread_local_i + i * n];
				}
			}
		}
		__syncthreads();

		for (size_t thread_local_i = 0; thread_local_i < TM; ++thread_local_i)
		{
			for (size_t thread_local_j = 0; thread_local_j < TN; ++thread_local_j)
			{
				PLS_PRAGMA_UNROLL
					PLS_FOR_I(slide_window)
				{
					tmp[thread_local_i][thread_local_j] += SA[threadIdx.x * TM + thread_local_i][i] * SB[i][threadIdx.y * TN + thread_local_j];
				}
			}
		}
		__syncthreads();
	}

	for (size_t thread_local_i = 0; thread_local_i < TM; ++thread_local_i)
	{
		for (size_t thread_local_j = 0; thread_local_j < TN; ++thread_local_j)
		{
			c[(row + thread_local_i) * n + col + thread_local_j] = tmp[thread_local_i][thread_local_j];
		}
	}
}

KERNEL_FUNC void gemm_block_tiled(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{

	size_t y_id = blockIdx.x * blockDim.x + threadIdx.x; // transpose block
	size_t x_id = blockIdx.y * blockDim.y + threadIdx.y;

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

HOST_FUNC cudaError_t Polaris::gemm_naive_kernel(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	constexpr Label dimx = 32;
	constexpr Label dimy = 32;
	dim3 dimBlock(dimx, dimy);
	dim3 dimGrid(
		(n + dimBlock.x - 1) / dimBlock.x,
		(m + dimBlock.y - 1) / dimBlock.y);
	gemm_naive << < dimGrid, dimBlock >> > (m, k, n, a, b, c);
	cudaError_t launchError = cudaPeekAtLastError();
	cudaError_t syncError = cudaDeviceSynchronize();
	return (launchError != cudaSuccess) ? launchError : syncError;
}

HOST_FUNC cudaError_t Polaris::gemm_smem_kernel(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	constexpr Label dimx = 32;
	constexpr Label dimy = 32;
	constexpr Label window_size = 32;
	dim3 dimBlock(dimx, dimy);
	dim3 dimGrid(
		(m + dimBlock.x - 1) / dimBlock.x,
		(n + dimBlock.y - 1) / dimBlock.y);
	gemm_smem<dimx, dimy, window_size> << < dimGrid, dimBlock >> > (m, k, n, a, b, c);
	cudaError_t launchError = cudaPeekAtLastError();
	cudaError_t syncError = cudaDeviceSynchronize();
	return (launchError != cudaSuccess) ? launchError : syncError;
}

HOST_FUNC cudaError_t Polaris::gemm_smem_tiled_kernel(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	constexpr Label dimx = 32;
	constexpr Label dimy = 32;
	constexpr Label window_size = 32;
	dim3 dimBlock(dimx, dimy);
	dim3 dimGrid(
		(m + dimBlock.x - 1) / dimBlock.x,
		(n + dimBlock.y - 1) / dimBlock.y);
	gemm_smem_tiled<dimx, dimy, window_size, 4, 4> << < dimGrid, dimBlock >> > (m, k, n, a, b, c);
	cudaError_t launchError = cudaPeekAtLastError();
	cudaError_t syncError = cudaDeviceSynchronize();
	return (launchError != cudaSuccess) ? launchError : syncError;
}

HOST_FUNC cudaError_t Polaris::gemm_block_tiled_kernel(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	return cudaSuccess;
}
