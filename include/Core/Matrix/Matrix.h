#ifndef PHYSICSENGINE_MATRIX_H
#define PHYSICSENGINE_MATRIX_H

// implementation of a static wrap of array

#include "Polaris.h"
#include <memory>
#include "Core/Storage/Storage.h"
#include "Core/Memory/allocator_CUDA.h"

namespace Polaris
{
	template <typename T, Platform deviceType = Platform::CPU,
        typename Allocator = typename std::conditional<deviceType == Platform::CPU,
        std::allocator<T>,
        cuda_utils::CudaAllocator<T>>::type>
	class Matrix
	{
    public:
        using ElementType = T;
        using AllocatorType = Allocator;
        using DataType = Storage<ElementType, deviceType, AllocatorType>;
        static constexpr Platform platform = deviceType;
        static constexpr Platform PlatformInfo()
        {
            return platform;
        }

    private:
        size_t _m;
        size_t _n;
        std::shared_ptr<DataType> _dataPtr;

    public:
        const Label m() const
        {
            return _m;
        }

        const Label n() const
        {
            return _n;
        }

        void Reshape(size_t m, size_t n)
        {
            auto currentDataSize = _dataPtr->Size();
            if (currentDataSize < m * n)
            {
                _dataPtr->Resize(m * n);
            }
            _m = m;
            _n = n;
        }

        HOST_FUNC Matrix() 
            :
            _m(0),
            _n(0),
            _dataPtr(std::make_shared<DataType>())
        {}

        HOST_FUNC Matrix(size_t m, size_t n)
            :
            _m(m),
            _n(n),
            _dataPtr(std::make_shared<DataType>(_m * _n))
        {
            _dataPtr->Resize(_m * _n);
            _dataPtr->FillZero();
        }

        HOST_FUNC Matrix(const Matrix& mat)
            :
            _m(mat._m),
            _n(mat._n),
            _dataPtr(std::make_shared<DataType>(*mat._dataPtr))
        {
            PLS_WARN("copy matrix of size {}-{} might be expensive", _m, _n);
        }

        HOST_FUNC Matrix(Matrix&& mat) noexcept
            :
            _m(mat._m),
            _n(mat._n),
            _dataPtr(std::move(mat._dataPtr))
        {}

        HOST_FUNC Matrix& operator=(const Matrix& mat) 
        {
            if (this != &mat) 
            {
                PLS_WARN("copy matrix of size {}-{} might be expensive", _m, _n);
                _m = mat._m;
                _n = mat._n;
                _dataPtr = std::make_shared<DataType>(*mat._dataPtr);
            }
            return *this;
        }

        HOST_FUNC Matrix& operator=(Matrix&& mat) noexcept {
            if (this != &mat) {
                _m = mat._m;
                _n = mat._n;
                _dataPtr = std::move(mat._dataPtr);
            }
            return *this;
        }

        HOST_FUNC virtual ~Matrix()
        {}

        HOST_FUNC T& operator()(size_t row, size_t col) {
            static_assert(platform == Platform::CPU, "direct access of device data is not allowed");
            assert(row < _m && col < _n && "Index out of bounds");
            return _dataPtr->At(row * _m + col);
        }

        HOST_FUNC const T& operator()(size_t row, size_t col) const {
            static_assert(platform == Platform::CPU, "direct access of device data is not allowed");
            assert(row < _m && col < _n && "Index out of bounds");
            return _dataPtr->At(row * _m + col);
        }

        // raw data
        HOST_FUNC auto Data() {
            return _dataPtr;
        }

        HOST_FUNC const auto Data() const {
            return _dataPtr;
        }

        // raw pointer
        HOST_FUNC T* RawData() {
            return _dataPtr->Data();
        }

        HOST_FUNC const T* RawData() const {
            return _dataPtr->Data();
        }

        HOST_FUNC void ClearData() {
            return _dataPtr->FillZero();
        }

        template <Platform OtherDevice, typename OtherAllocator>
        HOST_FUNC void Transfer(const Matrix<T, OtherDevice, OtherAllocator>& src) // enable cross platform copy
        {
            _m = src.m();
            _n = src.n();
            (*_dataPtr).Transfer(*(src.Data()));
        }
	};
}

template <typename T, Polaris::Platform deviceType = Polaris::Platform::CPU,
    typename Allocator = typename std::conditional<deviceType == Polaris::Platform::CPU,
    std::allocator<T>,
    cuda_utils::CudaAllocator<T>>::type>
HOST_FUNC std::ostream& operator<<(std::ostream& os, const Polaris::Matrix<T, deviceType, Allocator>& mat) {
    static_assert(deviceType == Polaris::Platform::CPU, "direct access of device data for output is not allowed");
    for (size_t i = 0; i < mat.m(); ++i) {
        os << "[ ";
        for (size_t j = 0; j < mat.n(); ++j) {
            os << mat(i, j) << " ";
        }
        os << "]" << std::endl;
    }
    return os;
}

#endif
