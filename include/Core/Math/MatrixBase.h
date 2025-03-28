#ifndef PHYSICSENGINE_MATRIX_BASE_H
#define PHYSICSENGINE_MATRIX_BASE_H

#include "Polaris.h"
#include "Core/Storage/Storage.h"

namespace Polaris
{
	template <typename T, Platform deviceType = Platform::CPU,
		typename Allocator = typename std::conditional<deviceType == Platform::CPU,
		std::allocator<T>,
		cuda_utils::CudaAllocator<T>>::type>
		class DynamicMatrixBase
	{
	private:
		Storage<T, Platform, Allocator> _data;
		Label _m;
		Label _n;

	public:
		DynamicMatrixBase()
			:
			_data(4),
			_m(0),
			_n(0)
		{

		}
	};

}

#endif