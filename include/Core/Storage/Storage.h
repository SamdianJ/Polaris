#ifndef PLS_STORAGE_H
#define PLS_STORAGE_H

#include <iostream>
#include <cassert>
#include <memory>
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
		static constexpr Platform PlatformInfo()
		{
			return platform;
		}
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
		class iterator {
		private:
			T* _ptr;
		public:
			HOST_FUNC iterator() 
				: 
				_ptr(nullptr) 
			{}

			HOST_FUNC iterator(T* p)
				: 
				_ptr(p) 
			{}

			HOST_FUNC iterator& operator++()
			{
				++_ptr;
				return *this;
			}

			HOST_FUNC iterator& operator--()
			{
				++_ptr;
				return *this;
			}

			HOST_FUNC iterator operator++(PlsInt32)
			{
				iterator ret(*this);
				++_ptr;
				return ret;
			}

			HOST_FUNC iterator operator--(PlsInt32)
			{
				iterator ret(*this);
				--_ptr;
				return ret;
			}

			HOST_FUNC T& operator*() {
				if constexpr (deviceType == Platform::CUDA) {
					static_assert(false, "Direct access to CUDA memory is not allowed!");
				}
				return *_ptr;
			}

			HOST_FUNC
				bool operator==(const iterator& other) const {
				return _ptr == other._ptr;
			}

			HOST_FUNC
				bool operator!=(const iterator& other) const {
				return _ptr != other._ptr;
			}
		};

		// constructor
		HOST_FUNC explicit Storage(size_t initialCapacity = 4, const Allocator& allocator = Allocator())
			: _size(0), _capacity(initialCapacity), _allocator(allocator) {
			_data = _allocator.allocate(_capacity);
		}

		// disallow cross platform copy
		HOST_FUNC Storage(const Storage& other)
		{
			static_assert(platform == other.platform, "Cross-platform copy not allowed!");

			// 复制元数据
			_size = other._size;
			_capacity = other._capacity;
			_allocator = other._allocator;

			_data = _allocator.allocate(_capacity);
			if (!_data || !other._data) {
				PLS_ERROR("Memory allocation failed!");
			}
			if constexpr (platform == Platform::CPU)
			{
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
			else
			{
#ifdef __CUDACC__
				CUDA_CHECK(cudaMemcpy(_data, other.Data(), _size * sizeof(T), cudaMemcpyDeviceToDevice));
				CUDA_CHECK(cudaDeviceSynchronize());
#else
				static_assert(deviceType != Platform::CUDA, "CUDA support is disabled!");
#endif
			}
		}

		Storage(Storage&& other) noexcept
			: _data(other._data),
			_size(other._size),
			_capacity(other._capacity),
			_allocator(std::move(other._allocator))
		{
			other._data = nullptr;
			other._size = 0;
			other._capacity = 0;
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
			static_assert(deviceType == Platform::CPU, "push back on device memory is not allowed");
			if (_size == _capacity) _expand(_capacity * 2);
			_data[_size++] = value;
		}

		HOST_FUNC void PopBack()
		{
			static_assert(deviceType == Platform::CPU, "pop back on device memory is not allowed");
			assert(_size > 0 && "Vector is empty!");
			--_size;
		}

		HOST_FUNC T& At(size_t index)
		{
			static_assert(deviceType == Platform::CPU, "Direct visit of device memory is not allowed");
			assert(index < _size && "Index out of bounds!");
			return reinterpret_cast<T&>(_data[index]);
		}

		HOST_FUNC const T& At(size_t index) const {
			static_assert(deviceType == Platform::CPU, "Direct visit of device memory is not allowed");
			assert(index < _size && "Index out of bounds!");
			return reinterpret_cast<const T&>(_data[index]);
		}

		HOST_FUNC T& operator[](size_t index)
		{
			static_assert(deviceType == Platform::CPU, "Direct visit of device memory is not allowed");
			assert(index < _size && "Index out of bounds!");
			return reinterpret_cast<T&>(_data[index]);
		}

		HOST_FUNC const T& operator[](size_t index) const {
			static_assert(deviceType == Platform::CPU, "Direct visit of device memory is not allowed");
			assert(index < _size && "Index out of bounds!");
			return reinterpret_cast<const T&>(_data[index]);
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
		HOST_FUNC void Clear() {
			_size = 0;
		}

		HOST_FUNC void ClearMemory() {
			_allocator.deallocate(_data);
			_capacity = 0;
			_size = 0;
			_data = nullptr;
		}

		HOST_FUNC T* Data() {
			return reinterpret_cast<T*> (_data);
		}

		HOST_FUNC const T* Data() const {
			return reinterpret_cast<const T*> (_data);
		}

		HOST_FUNC void FillZero() {
			if constexpr (platform == Platform::CUDA)
			{
				if (_size < 1)
					return;

				CUDA_CHECK(cudaMemset(_data, 0, _size * sizeof(T)));
			}
			else
			{
				std::memset(_data, 0, _size * sizeof(T));
			}
		}

		HOST_FUNC void FillInfi() {
			if constexpr (platform == Platform::CUDA)
			{
				if (_size < 1)
					return;

				CUDA_CHECK(cudaMemset(_data, 0x3f3f3f3f, _size * sizeof(T)));
			}
			else
			{
				std::memset(_data, 0x3f3f3f3f, _size * sizeof(T));
			}
		}

		HOST_FUNC void Fill(T v) {
			if constexpr (platform == Platform::CUDA) // TO DO should implemented on kernel
			{
				PLS_WARN("[Performance Warning] fill values on device is not recommended!");
				Storage<T, Platform::CPU> tmp;
				tmp.Resize(_size);
				tmp.Fill(v);
				Transfer(tmp);
			}
			else
			{
				PLS_FOR_I(_size)
				{
					_data[i] = v;
				}
			}
		}

		// 获取大小
		HOST_FUNC void Print(bool printData = false) const
		{
			static_assert(deviceType == Platform::CPU, "print of device data is not allowed");
			std::cout << "data Size: " << _size << "\tmemory occupation: " << Scalar(_capacity * sizeof(T)) / 1024.0 / 1024.0 << "Mbs" << std::endl;
			if (printData)
			{
				std::cout << "Data: [" << std::endl;
				PLS_FOR_I(_size)
				{
					std::cout << "index " << i << "\tdata: " << _data[i] << std::endl;
				}
				std::cout << "]" << std::endl;
			}
		}

		// 迭代器支持
		HOST_FUNC auto Begin()
		{
			return iterator(_data);
		}

		HOST_FUNC auto End() 
		{
			return iterator(reinterpret_cast<T*> (_data + _size));
		}
	};
}

#endif