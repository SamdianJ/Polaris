#ifndef PLS_STORAGE_H
#define PLS_STORAGE_H

#include <iostream>
#include <cassert>
#include <memory>
#include "PolarisMacro.h"
#include "Platform/Platforms.h"
#include "Core/Memory/allocator_CUDA.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace Polaris
{
	template<typename T, Platform deviceType = Platform::CPU,
		typename Allocator = typename std::conditional<deviceType == Platform::CPU,
		std::allocator<T>,
		cuda_utils::CudaAllocator<T>>::type>
		class Storage
	{
	public:
		static constexpr Platform platform = deviceType;
	private:
		T* _data;        // raw data
		size_t _size;    
		size_t _capacity; 
		Allocator _allocator; // allocator

		// 扩展容量
		HOST_FUNC void _expand(size_t newCapacity) {
			if (newCapacity <= _capacity) return;

			T* newData = _allocator.allocate(newCapacity);
			if constexpr (deviceType == Platform::CUDA)
			{
#ifdef __CUDACC__
				CUDA_CHECK(cudaMemcpy(newData, _data, _size * sizeof(T), cudaMemcpyDeviceToDevice));
				CUDA_CHECK(cudaDeviceSynchronize());
#else
				static_assert(deviceType != Platform::CUDA, "CUDA support is disabled!");
#endif
			}
			else {
				// 主机内存拷贝
				for (size_t i = 0; i < _size; ++i) {
					newData[i] = std::move(_data[i]);
				}
			}
			_allocator.deallocate(_data, _capacity);
			_data = newData;
			_capacity = newCapacity;
		}

	public:
		// iterator
		class Iterator {
		private:
			T* _ptr;
		public:
			HOST_FUNC explicit Iterator(T* p) : _ptr(p) {}

			HOST_FUNC T& operator*() {
				if constexpr (deviceType == Platform::CUDA) {
					static_assert(false, "Direct access to CUDA memory is not allowed!");
				}
				return *_ptr;
			}
			HOST_FUNC Iterator& operator++() { ++_ptr; return *this; }
			HOST_FUNC bool operator!=(const Iterator& other) const { return _ptr != other._ptr; }
		};

#ifdef __CUDACC__
#include <thrust/device_ptr.h>
		class CudaIterator {
		private:
			thrust::device_ptr<T> _ptr;
		public:
			HOST_FUNC explicit CudaIterator(T* p) : _ptr(thrust::device_pointer_cast(p)) {}
			HOST_FUNC thrust::device_ptr<T> operator++() { return ++_ptr; }
			HOST_FUNC bool operator!=(const CudaIterator& other) const { return _ptr != other._ptr; }
		};
#endif

		// constructor
		HOST_FUNC explicit Storage(size_t initialCapacity = 4, const Allocator& allocator = Allocator())
			: _size(0), _capacity(initialCapacity), _allocator(allocator) {
			_data = _allocator.allocate(_capacity);
		}

		// cpu copy is allowed
		template <Platform OtherDevice, typename OtherAllocator,
			typename = std::enable_if_t<OtherDevice == Platform::CPU>>
			HOST_FUNC Storage(const Storage<T, OtherDevice, OtherAllocator>& other)
			: _size(other._size),
			_capacity(other._capacity),
			_allocator(other._allocator)
		{
			// 只有当 other 平台为 CPU 时，才能进入该分支
			_data = _allocator.allocate(_capacity);
			// 对于 trivially copyable 的类型可以使用 memcpy，否则建议逐个构造
			if constexpr (std::is_trivially_copyable_v<T>) {
				memcpy(_data, other._data, _size * sizeof(T));
			}
			else {
				for (size_t i = 0; i < _size; ++i) {
					new(&_data[i]) T(other._data[i]);
				}
			}
		}

		// disallow cross platform copy
		HOST_FUNC Storage(const Storage& other) = delete;

		// platform Info
		HOST_FUNC Platform PlatformInfo() const
		{
			return platform;
		}

		// deconstructor
		HOST_FUNC ~Storage() {
			_allocator.deallocate(_data, _capacity);
		}

		HOST_FUNC void Reserve(size_t newCapacity) {
			_expand(newCapacity);
		}

		HOST_FUNC void Resize(size_t newSize) {
			if (newSize > _capacity) {
				_expand(newSize);
			}
			_size = newSize;
		}

		HOST_FUNC void PushBack(const T& value)
		{
			static_assert(constexpr(deviceType == Platform::CPU), "push back on device memory is not allowed");
			if (_size == _capacity) _expand(_capacity * 2);
			_data[_size++] = value;
		}

		HOST_FUNC void PopBack()
		{
			static_assert(constexpr(deviceType == Platform::CPU), "pop back on device memory is not allowed");
			assert(_size > 0 && "Vector is empty!");
			--_size;
		}

		HOST_FUNC T& operator[](size_t index)
		{
			static_assert(constexpr(deviceType == Platform::CPU), "Direct visit of device memory is not allowed");
			assert(index < _size && "Index out of bounds!");
			return _data[index];
		}

		HOST_FUNC const T& operator[](size_t index) const {
			static_assert(constexpr(deviceType == Platform::CPU), "Direct visit of device memory is not allowed");
			assert(index < _size && "Index out of bounds!");
			return _data[index];
		}

		template <Platform OtherDevice, typename OtherAllocator>
		HOST_FUNC void Transfer(const Storage<T, OtherDevice, OtherAllocator>& src) // enable cross platform copy
		{
			// resize
			this->Resize(src.Size());

			// get raw pointers
			const T* srcData = src.Data();
			T* dstData = this->Data();

			if constexpr (deviceType == Platform::CPU && OtherDevice == Platform::CPU) 
			{
				// std::copy
				std::copy(srcData, srcData + src.Size(), dstData);
			}
#ifdef __CUDACC__
			else {
				// data transfer type
				cudaMemcpyKind kind;
				if constexpr (deviceType == Platform::CUDA && OtherDevice == Platform::CPU) {
					kind = cudaMemcpyHostToDevice; // CPU -> CUDA
				}
				else if constexpr (deviceType == Platform::CPU && OtherDevice == Platform::CUDA) {
					kind = cudaMemcpyDeviceToHost; // CUDA -> CPU
				}
				else if constexpr (deviceType == Platform::CUDA && OtherDevice == Platform::CUDA) {
					kind = cudaMemcpyDeviceToDevice; // CUDA -> CUDA
				}
				else {
					static_assert(deviceType == Platform::CPU && OtherDevice == Platform::CPU,
						"Unsupported platform combination!");
				}

				// cuda copy
				CUDA_CHECK(cudaMemcpy(
					dstData,
					srcData,
					src.Size() * sizeof(T),
					kind
				));
				CUDA_CHECK(cudaDeviceSynchronize()); // 确保拷贝完成
			}
#else
			else 
			{
				// disable cross platform copy if no CUDA is found
				static_assert(deviceType == Platform::CPU && OtherDevice == Platform::CPU,
					"CUDA operations require CUDA compiler support!");
			}
#endif
		}

#ifdef __CUDACC__
		template <Platform OtherDevice, typename OtherAllocator>
		HOST_FUNC Storage<T, Platform::CPU> LoadToHost(const Storage<T, OtherDevice, OtherAllocator>& src)
		{
			static_assert(OtherDevice == Platform::CUDA, "L2H for non CUDA storage makes no sense, use Transfer instead");	
			Storage<T, Platform::CPU> buffer;
			buffer.Transfer(src);
			return buffer;
		}

		template <Platform OtherDevice, typename OtherAllocator>
		HOST_FUNC Storage<T, Platform::CUDA> LoadToDevice(const Storage<T, OtherDevice, OtherAllocator>& src)
		{
			static_assert(OtherDevice == Platform::CPU, "H2D for non CUDA storage makes no sense, use Transfer instead");
			Storage<T, Platform::CUDA> buffer;
			buffer.Transfer(src);
			return buffer;
		}
#endif

		// 获取大小
		HOST_FUNC size_t Size() const { return _size; }

		// 获取容量
		HOST_FUNC size_t Capacity() const { return _capacity; }

		// 清空Vector
		void Clear() {
			_size = 0;
		}

		// 迭代器支持
		HOST_FUNC auto Begin()
		{ 
			if constexpr (deviceType == Platform::CUDA)
			{
#ifdef __CUDACC__
				return CudaIterator(_data);
#else
				static_assert(deviceType != Platform::CUDA, "device iterator is disableed as CUDA support is disabled!")
#endif
			}
			else
			{
				return Iterator(_data);
			}
		}
		HOST_FUNC auto End() {
			if constexpr (deviceType == Platform::CUDA)
			{
#ifdef __CUDACC__
				return CudaIterator(_data + _size);
#else
				static_assert(deviceType != Platform::CUDA, "device iterator is disableed as CUDA support is disabled!")
#endif
			}
			else
			{
				return Iterator(_data + _size);
			}
		}
	};
}

#endif