#include "Test/Core/MatrixTest.h"
#include <iostream>
#include "Core/Timer/Timer.h"
#include <random>
#include "Core/LinearAlgebra/gemm.h"

using namespace Polaris;

HOST_FUNC Matrix<Scalar, Platform::CPU> GetRandomMatrix(Label m, Label n, PlsTimer& timer)
{
	timer.HostStart("construct matrix");
	Matrix<Scalar, Platform::CPU> matrix(m, n);
	static std::random_device rd; 
	static std::mt19937 gen(rd()); 
	std::uniform_real_distribution<Scalar> dis(0.0, 1.0);
	PLS_FOR_I(m)
	{
		PLS_FOR_J(n)
		{
			matrix(i, j) = dis(gen);
		}
	}
	timer.HostStop("construct matrix");
	return matrix;
}

void Polaris::Device::MatrixTest()
{
	//static constexpr size_t M = 2048;
	//static constexpr size_t N = 2048;

	static constexpr size_t M = 1024;
	static constexpr size_t N = 1024;
	static constexpr size_t K = 1024;

	auto& timer = PlsTimer::GetInstance();

	auto matrix1 = GetRandomMatrix(M, N, timer);
	auto matrix2 = GetRandomMatrix(N, K, timer);

	Matrix<Scalar, Platform::CUDA> gmatrix1;
	gmatrix1.Transfer(matrix1);
	Matrix<Scalar, Platform::CUDA> gmatrix2;
	gmatrix2.Transfer(matrix2);

	sgemm<Platform::CPU, std::allocator<Scalar>> gemm;
	auto& cpuRes = gemm(matrix1, matrix2, &timer);
	sgemm<Platform::CUDA, cuda_utils::CudaAllocator<Scalar>> ggemm;
	auto& gpuRes = ggemm(gmatrix1, gmatrix2, &timer);

	Matrix<Scalar, Platform::CPU> gpuResToCpu;
	gpuResToCpu.Transfer(gpuRes);

	Scalar eps = 1e-10f;
	bool isSame = true;
	PLS_FOR_I(M)
	{
		PLS_FOR_J(K)
		{
			bool elementTest = abs(cpuRes(i, j) - gpuResToCpu(i, j)) < eps;
			if (!elementTest)
			{
				PLS_WARN("{}, {} element dismatch: {}-{},{}", i, j, cpuRes(i, j), gpuResToCpu(i, j),abs(cpuRes(i, j) - gpuResToCpu(i, j)));
			}
			isSame = isSame && elementTest;
		}
	}
	if (isSame)
	{
		PLS_INFO("gemm test finish");
	}

	timer.PrintAll();
}