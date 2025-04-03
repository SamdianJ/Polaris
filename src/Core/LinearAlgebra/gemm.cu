#include "Core/LinearAlgebra/gemm.cuh"
#include "Platform/CUDA/kernelLaunch.h"

template<typename T, Polaris::Label TILE_M, Polaris::Label TILE_K>
DEVICE_FUNC void load_gmem_tile_to_smem(T* gmem, size_t g_index, T* smem)
{
	smem_m = threadIdx.x % TILE_K;
	smem_n = threadIdx.x / TILE_K;

	PLS_FOR_I(TILE_M)
	{
		PLS_PRAGMA_UNROLL
			PLS_FOR_J(TILE_K)
		{
			smem[smem_m * TILE_K + smem_n] = (gmem + g_index * TILE_M * TILE_K)[smem_m * TILE_K + smem_n];
		}
	}
}

template<typename T>
DEVICE_FUNC void mma4x4(T* a, T* b, T* c)
{
	PLS_FOR_I(4)
	{
		PLS_FOR_J(4)
		{
			T tmp = T(0.0); // register
			PLS_PRAGMA_UNROLL 4
			PLS_FOR_K(4)
			{
				tmp += a[i * 4 + k] + b[k * 4 + j];
			}
			c[i * 4 + j] = tmp;
		}
	}
}

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
	constexpr size_t slide_window = SW;
	size_t num_slide_window = k / slide_window;

	__shared__ Scalar SA[BM][slide_window]; // SA[row][col]
	__shared__ Scalar SB[slide_window][BN];

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
	size_t y_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t x_id = blockIdx.y * blockDim.y + threadIdx.y;

	Scalar tmp[TM][TN] = { Scalar(0) };
	constexpr size_t slide_window = SW;
	size_t num_slide_window = k / slide_window;

	__shared__ Scalar SA[BM * TM][SW]; // SA[row][col]
	__shared__ Scalar SB[SW][BN * TN];

	if (TM * x_id >= m || TN * y_id >= n)
		return;

	for (size_t slide_id = 0; slide_id < num_slide_window; ++slide_id)
	{ 
		for (size_t thread_local_i = 0; thread_local_i < TM; ++thread_local_i)
		{
			// load to smem
			size_t a_row = (blockIdx.y * TM + thread_local_i) * BM + threadIdx.y;
			size_t a_col = threadIdx.x + slide_id * slide_window;
			if (threadIdx.x < slide_window)
			{
				SA[threadIdx.y + BM * thread_local_i][threadIdx.x] = a[a_row * k + a_col];
			}
		} 
		for (size_t thread_local_i = 0; thread_local_i < TN; ++thread_local_i)
		{
			size_t b_row = threadIdx.y + slide_id * slide_window;
			size_t b_col = threadIdx.x + (blockIdx.x * TN + thread_local_i) * BN;
			if (threadIdx.y < slide_window)
			{
				SB[threadIdx.y][thread_local_i * BN + threadIdx.x] = b[b_row * n + b_col];
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
					tmp[thread_local_i][thread_local_j] += SA[threadIdx.y + thread_local_i * BM][i] * SB[i][threadIdx.x + thread_local_j * BN];
				}
			}
		}
		__syncthreads();
	}

	for (size_t thread_local_i = 0; thread_local_i < TM; ++thread_local_i)
	{
		for (size_t thread_local_j = 0; thread_local_j < TN; ++thread_local_j)
		{
			c[((blockIdx.y * TM + thread_local_i) * BM + threadIdx.y) * n + threadIdx.x + (blockIdx.x * TN + thread_local_j) * BN] = tmp[thread_local_i][thread_local_j];
		}
	}
}

template <typename size_t BM, typename size_t BN, typename size_t TILE_M, typename size_t TILE_K, typename size_t TILE_N>
KERNEL_FUNC void gemm_smem_tiled_1D(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	__shared__ Scalar smemA[TILE_M * TILE_K];
	__shared__ Scalar smemB[TILE_N * TILE_K];

	size_t y_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t x_id = blockIdx.y * blockDim.y + threadIdx.y;
	if (TILE_N * x_id >= m || TILE_M * y_id >= n)
		return;

	Scalar tmp[TILE_M][TILE_N] = { Scalar(0) };

	int tid = threadIdx.y * blockDim.x + threadIdx.x;

	int smem_m = tid % TILE_K;
	int smem_n = tid / TILE_K;

	Scalar tmp

	for (Label i = 0; i < k; i += TILE_K)
	{
		__syncthreads();
		//load_gmem_tile_to_smem
		smemA[smem_m * blockDim.x + smem_n] = a[];
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
	constexpr Label window_size = 16;
	constexpr Label tilex = 2;
	constexpr Label tiley = 2;
	dim3 dimBlock(dimx, dimy);
	dim3 dimGrid(
		(m + tilex * dimBlock.x - 1) / (tilex * dimBlock.x),
		(n + tiley * dimBlock.y - 1) / (tiley * dimBlock.y));
	gemm_smem_tiled<dimx, dimy, window_size, tilex, tiley> << < dimGrid, dimBlock >> > (m, k, n, a, b, c);
	cudaError_t launchError = cudaPeekAtLastError();
	cudaError_t syncError = cudaDeviceSynchronize();
	return (launchError != cudaSuccess) ? launchError : syncError;
}

HOST_FUNC cudaError_t Polaris::gemm_smem_tiled_1D_kernel(size_t m, size_t k, size_t n, const Scalar* a, const Scalar* b, Scalar* c)
{
	constexpr Label dimx = 32;
	constexpr Label dimy = 32;
	constexpr Label tile_size = 128;
	constexpr Label tilex = 4;
	constexpr Label tiley = 4;
	dim3 dimBlock(dimx, dimy);
	dim3 dimGrid(
		(m + tilex * dimBlock.x - 1) / (tilex * dimBlock.x),
		(n + tiley * dimBlock.y - 1) / (tiley * dimBlock.y));
	gemm_smem_tiled_1D<dimx, dimy, tile_size, tilex, tiley> << < dimGrid, dimBlock >> > (m, k, n, a, b, c);
	cudaError_t launchError = cudaPeekAtLastError();
	cudaError_t syncError = cudaDeviceSynchronize();
	return (launchError != cudaSuccess) ? launchError : syncError;
}
