#ifndef PHYSICSENGINE_ARRAY_H
#define PHYSICSENGINE_ARRAY_H

// implementation of a static wrap of array

#include "Polaris.h"

namespace Polaris
{
	template <typename T, Label N>
	struct Array
	{
		using StorageType = T;
		using ElementType = T;

		static constexpr size_t kStorageElements = N;
		static constexpr size_t kElements = N;

		typedef T value_type;
		typedef size_t size_type;
		typedef ptrdiff_t difference_type;
		typedef value_type& reference;
		typedef value_type const& const_reference;
		typedef value_type* pointer;
		typedef value_type const* const_pointer;

		class iterator
		{
			T* _ptr;

		public:
			HOST_DEVICE_FUNC iterator()
				:
				_ptr(nullptr)
			{}

			HOST_DEVICE_FUNC iterator(T* ptr)
				:
				_ptr(ptr)
			{}

			HOST_DEVICE_FUNC iterator& operator++()
			{
				++_ptr;
				return *this;
			}

			HOST_DEVICE_FUNC iterator& operator--()
			{
				--_ptr;
				return *this;
			}

			HOST_DEVICE_FUNC iterator operator++(PlsInt32)
			{
				iterator ret(*this);
				++_ptr;
				return ret;
			}

			HOST_DEVICE_FUNC iterator operator--(PlsInt32)
			{
				iterator ret(*this);
				--_ptr;
				return ret;
			}

			HOST_DEVICE_FUNC T& operator*() const
			{
				return *_ptr;
			}

			HOST_DEVICE_FUNC
				bool operator==(const iterator& other) const {
				return _ptr == other._ptr;
			}

			HOST_DEVICE_FUNC
				bool operator!=(const iterator& other) const {
				return _ptr != other._ptr;
			}
		};

		StorageType _data[kElements];

		HOST_DEVICE_FUNC void clear()
		{
			fill(T(0));
		}

		HOST_DEVICE_FUNC reference at(size_type i)
		{
			return reinterpret_cast<reference> (_data[i]);
		}

		HOST_DEVICE_FUNC const_reference at(size_type i) const
		{
			return reinterpret_cast<const_reference> (_data[i]);
		}

		HOST_DEVICE_FUNC reference operator[](size_type i)
		{
			return reinterpret_cast<reference> (_data[i]);
		}

		HOST_DEVICE_FUNC const_reference operator[](size_type i) const
		{
			return reinterpret_cast<const_reference> (_data[i]);
		}

		HOST_DEVICE_FUNC reference font()
		{
			return reinterpret_cast<reference> (_data[0]);
		}

		HOST_DEVICE_FUNC const_reference font() const
		{
			return reinterpret_cast<const_reference> (_data[0]);
		}

		HOST_DEVICE_FUNC reference back()
		{
			return reinterpret_cast<reference> (_data[kElements - 1]);
		}

		HOST_DEVICE_FUNC const_reference back() const
		{
			return reinterpret_cast<const_reference> (_data[kElements - 1]);
		}

		HOST_DEVICE_FUNC pointer data()
		{
			return reinterpret_cast<pointer> (_data);
		}

		HOST_DEVICE_FUNC const_pointer data() const
		{
			return reinterpret_cast<const_pointer> (_data);
		}

		HOST_DEVICE_FUNC pointer raw_data()
		{
			return reinterpret_cast<pointer> (_data);
		}

		HOST_DEVICE_FUNC const_pointer raw_data() const
		{
			return reinterpret_cast<const_pointer> (_data);
		}

		HOST_DEVICE_FUNC bool empty()
		{
			return !kElements;
		}

		HOST_DEVICE_FUNC size_type size()
		{
			return kElements;
		}

		HOST_DEVICE_FUNC size_type max_size()
		{
			return kElements;
		}

		HOST_DEVICE_FUNC void fill(const T& value)
		{
			PLS_PRAGMA_UNROLL 
			PLS_FOR_I(PlsInt32(kElements))
			{
				_data[i] = value;
			}
		}

		HOST_DEVICE_FUNC iterator begin()
		{
			return iterator(_data);
		}

		HOST_DEVICE_FUNC iterator end()
		{
			return iterator(reinterpret_cast<pointer>(_data + kStorageElements));
		}
	};

	//////////////////////////////////////////////
	//Factories
	//////////////////////////////////////////////
	template <typename T>
	HOST_DEVICE_FUNC Array<T, 1> MakeArrayT(T x)
	{
		return {x};
	}

	template <typename T>
	HOST_DEVICE_FUNC Array<T, 2> MakeArray2T(T x, T y)
	{
		return { x, y };
	}

	template <typename T>
	HOST_DEVICE_FUNC Array<T, 3> MakeArray3T(T x, T y, T z)
	{
		return { x, y, z };
	}

	template <typename T>
	HOST_DEVICE_FUNC Array<T, 4> MakeArray3T(T x, T y, T z, T w)
	{
		return { x, y, z, w };
	}
}

#endif
