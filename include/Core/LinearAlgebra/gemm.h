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
#include <string>

namespace Polaris 
{
	enum GemmMethod
	{
		kNaive,
		kSmem,
		kSmemTiled,
		kCutlass
	};

	static std::string GemmMethodToString(GemmMethod method)
	{
		switch (method)
		{
		case kNaive:
			return "[gemm] naive";
		case kSmem:
			return "[gemm] SharedMemory";
		case kSmemTiled:
			return "[gemm] SharedMemory Tiled";
		default:
			break;
		}
	}

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
		PlsTimer* _timer;

	public:
		explicit HOST_FUNC sgemm(PlsTimer* timer = nullptr)
			:
			_m(0),
			_n(0),
			_k(0),
			_timer(timer)
		{

		}

		HOST_FUNC sgemm(size_t m, size_t n, size_t k, PlsTimer* timer = nullptr)
			:
			_work(m, k),
			_m(m),
			_n(n),
			_k(k),
			_timer(timer)
		{

		}

		HOST_FUNC virtual ~sgemm()
		{}

		HOST_FUNC auto& operator()(const Matrix<Scalar, platform>& a, const Matrix<Scalar, platform>& b)
		{
			auto m = a.m();
			auto n1 = a.n();
			auto n2 = b.m();
			auto k = b.n();
			assert(n1 == n2, "incompatible matrix size");
			_work.Reshape(m, k);
			_work.ClearData();

			if (_timer)
			{
				_timer->HostStart("[gemm] cpu");
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
			if (_timer)
			{
				_timer->HostStop("[gemm] cpu");
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
		PlsTimer* _timer;
		GemmMethod _method;

	public:
		HOST_FUNC sgemm(PlsTimer* timer = nullptr)
			:
			_m(0),
			_n(0),
			_k(0),
			_timer(timer),
			_method(GemmMethod::kNaive)
		{

		}

		HOST_FUNC sgemm(size_t m, size_t n, size_t k, PlsTimer* timer = nullptr)
			:
			_work(m, k),
			_m(m),
			_n(n),
			_k(k),
			_timer(timer),
			_method(GemmMethod::kNaive)
		{

		}

		HOST_FUNC virtual ~sgemm()
		{}

		HOST_FUNC auto& operator()(const Matrix<Scalar, platform>& a, const Matrix<Scalar, platform>& b, GemmMethod gemmMethod = GemmMethod::kNaive)
		{
			auto m = a.m();
			auto n1 = a.n();
			auto n2 = b.m();
			auto k = b.n();
			assert(n1 == n2, "incompatible matrix size");
			_work.Reshape(m, k);
			_work.ClearData();

#ifdef __CUDACC__
			if (_timer)
			{
				_timer->DeviceStart(GemmMethodToString(gemmMethod));
			}

			cudaError_t ret;
			if (gemmMethod == GemmMethod::kNaive)
			{
				ret = Polaris::gemm_naive_kernel(m, n1, k, a.RawData(), b.RawData(), _work.RawData());
			}
			else if (gemmMethod == GemmMethod::kSmem)
			{
				ret = Polaris::gemm_smem_kernel(m, n1, k, a.RawData(), b.RawData(), _work.RawData());
			}
			else if (gemmMethod == GemmMethod::kSmemTiled)
			{
				ret = Polaris::gemm_smem_tiled_kernel(m, n1, k, a.RawData(), b.RawData(), _work.RawData());
			}
			else
			{
				PLS_ERROR("unknow gemm method");
			}
			if (ret != cudaSuccess)
			{
				PLS_ERROR("kernel launch fails");
			}
			if (_timer)
			{
				_timer->DeviceStop(GemmMethodToString(gemmMethod));
			}
#endif
			return _work;
		}
	};

} // namespace PhysicsEngine

#endif // PHYSICSENGINE_MATH_MATRIX3_H
