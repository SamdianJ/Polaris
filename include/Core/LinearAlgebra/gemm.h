#ifndef PHYSICSENGINE_SGEMM_H
#define PHYSICSENGINE_SGEMM_H

#include <cmath>
#include <iostream>
#include "Polaris.h"
#include "Core/Matrix/Matrix.h"
#include "Core/Timer/Timer.h"
#ifdef __CUDACC__
#include "gemm.cuh"
#endif

namespace Polaris 
{
	template <Platform platform, typename Allocator> class sgemm;

	template <>
	class sgemm<Platform::CPU, std::allocator<Scalar>>
	{
	public:
		static constexpr Platform platform = Platform::CPU;
		static constexpr Platform PlatformInfo()
		{
			return platform;
		}

	private:
		Matrix<Scalar, platform, std::allocator<Scalar>> _work;
		size_t _m;
		size_t _n;
		size_t _k;

	public:
		HOST_FUNC sgemm()
			:
			_m(0),
			_n(0),
			_k(0)
		{

		}

		HOST_FUNC sgemm(size_t m, size_t n, size_t k)
			:
			_work(m, k),
			_m(m),
			_n(n),
			_k(k)
		{

		}

		HOST_FUNC virtual ~sgemm()
		{}

		HOST_FUNC auto& operator()(const Matrix<Scalar, platform>& a, const Matrix<Scalar, platform>& b, PlsTimer* timer = nullptr)
		{
			auto m = a.m();
			auto n1 = a.n();
			auto n2 = b.m();
			auto k = b.n();
			assert(n1 == n2, "incompatible matrix size");
			_work.Reshape(m, k);
			_work.ClearData();

			if (timer)
			{
				timer->HostStart("gemm");
			}
			PLS_FOR_I(m)
			{
				PLS_FOR_J(k)
				{
					Scalar res = 0.0f;
					for (size_t idx = 0; idx < n1; ++idx)
					{
						res += a(i, idx) * b(idx, j);
					}
					_work(i, j) = res;
				}
			}
			if (timer)
			{
				timer->HostStop("gemm");
			}
			return _work;
		}
	};

	template <>
	class sgemm<Platform::CUDA, cuda_utils::CudaAllocator<Scalar>>
	{
	public:
		static constexpr Platform platform = Platform::CUDA;
		static constexpr Platform PlatformInfo()
		{
			return platform;
		}

	private:
		Matrix<Scalar, platform, cuda_utils::CudaAllocator<Scalar>> _work;
		size_t _m;
		size_t _n;
		size_t _k;

	public:
		HOST_FUNC sgemm()
			:
			_m(0),
			_n(0),
			_k(0)
		{

		}

		HOST_FUNC sgemm(size_t m, size_t n, size_t k)
			:
			_work(m, k),
			_m(m),
			_n(n),
			_k(k)
		{

		}

		HOST_FUNC virtual ~sgemm()
		{}

		HOST_FUNC auto& operator()(const Matrix<Scalar, platform>& a, const Matrix<Scalar, platform>& b, PlsTimer* timer = nullptr)
		{
			auto m = a.m();
			auto n1 = a.n();
			auto n2 = b.m();
			auto k = b.n();
			assert(n1 == n2, "incompatible matrix size");
			_work.Reshape(m, k);
			_work.ClearData();

#ifdef __CUDACC__
			if (timer)
			{
				timer->DeviceStart("gemm");
			}
			auto ret = Polaris::gemm_kernel(m, n1, k, a.RawData(), b.RawData(), _work.RawData());
			if (ret != cudaSuccess)
			{
				PLS_ERROR("kernel launch fails");
			}
			if (timer)
			{
				timer->DeviceStop("gemm");
			}
#endif
			return _work;
		}
	};
} // namespace PhysicsEngine

#endif // PHYSICSENGINE_MATH_MATRIX3_H
