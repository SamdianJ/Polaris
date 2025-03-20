#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#include <cuda_runtime.h>
#include <limits>
#include <new>
#include <stdexcept>
#include "Platform/Platforms.h"
#include <iostream>
#include "Polaris.h"

namespace cuda_utils
{
	template<typename T>
	class CudaAllocator {
	public:
		using value_type = T;
		using pointer = T*;
		using const_pointer = const T*;
		using reference = T&;
		using const_reference = const T&;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;

		// rebind
		template<typename U>
		struct rebind {
			using other = CudaAllocator<U>;
		};

		CudaAllocator() noexcept {}

		template<typename U>
		CudaAllocator(const CudaAllocator<U>&) noexcept {}

		// allocate
		pointer allocate(size_type n, const void* hint = 0) {
#ifdef __CUDACC__
			if (n > std::numeric_limits<size_type>::max() / sizeof(T))
				throw std::bad_alloc();

			T* ptr = nullptr;
			CUDA_CHECK(cudaMalloc((void**)&ptr, n * sizeof(T)));
			PLS_WARN("Device memory allocated {} Mb", Scalar(n * sizeof(T)) / 1024 / 1024);
			return ptr;
#else
			static_assert(false, "CUDA is not activated!");
#endif
		}

		// deallocate
		void deallocate(pointer p, size_type n = -1) noexcept {
#ifdef __CUDACC__
			cudaFree(p);
#else
			static_assert(false, "CUDA is not activated!");
#endif
		}

#ifdef __CUDACC__
		using propagate_on_container_copy_assignment = std::true_type;
		using propagate_on_container_move_assignment = std::true_type;
		using propagate_on_container_swap = std::true_type;
#else
		using propagate_on_container_copy_assignment = std::false_type;
		using propagate_on_container_move_assignment = std::false_type;
		using propagate_on_container_swap = std::false_type;
#endif
	};


	template<typename T, typename U>
	bool operator==(const CudaAllocator<T>&, const CudaAllocator<U>&) { return true; }

	template<typename T, typename U>
	bool operator!=(const CudaAllocator<T>& a, const CudaAllocator<U>& b) { return !(a == b); }
}


#endif // CUDA_ALLOCATOR_H

