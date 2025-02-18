#ifndef AXMATRIX_H
#define AXMATRIX_H

#include "macro.h"
#include "dataType.h"
#include "assert.h"
#include <iostream>
#include "memory.h"

template <typename T>
class AxMatrix3x3
{
private:
    mutable T data_[9];

public:
    AxMatrix3x3()
    {
        memset(data_, 0, sizeof(T) * 9);
    }

    AxMatrix3x3(const AxMatrix3x3<T>& other)
    {
        memcpy(data_, other.data(), sizeof(T) * 9);
    }

    template <typename T1>
    AxMatrix3x3(AxMatrix3x3<T1> other)
    {
        AX_FOR_I(9)
        {
            data_[i] = T(other.data()[i]);
        }
    }

    T* data()
    {
        return data_;
    }

    T& operator()(AxUInt32 m, AxUInt32 n)
    {
        if (m > 2 || n > 2)
            std::cerr << "Invalid visit dimension" << std::endl;
        return data_[3 * m + n];
    }

    T operator()(AxUInt32 m, AxUInt32 n) const
    {
        if (m > 2 || n > 2)
            std::cerr << "Invalid visit dimension" << std::endl;
        return data_[3 * m + n];
    }

    T trace()
    {
        return data_[0] + data_[4] + data_[8];
    }

    T determinent()
    {
        T d1 = data_[0] * (data_[4] * data_[8] - data_[7] * data_[5]);
        T d2 = data_[1] * (data_[3] * data_[8] - data_[6] * data_[5]);
        T d3 = data_[2] * (data_[3] * data_[7] - data_[6] * data_[4]);
        return d1 + d3 - d2;
    }

    AxMatrix3x3<T> inverse()
    {
        T det = determinent();
        AxMatrix3x3<T> inv;
        inv(0,0) = data_[4] * data_[8] - data_[7] * data_[5];
        inv(0,1) = data_[2] * data_[7] - data_[1] * data_[8];
        inv(0,2) = data_[1] * data_[5] - data_[2] * data_[4];
        inv(1,0) = data_[5] * data_[6] - data_[3] * data_[8];
        inv(1,1) = data_[0] * data_[8] - data_[6] * data_[2];
        inv(1,2) = data_[3] * data_[2] - data_[0] * data_[5];
        inv(2,0) = data_[3] * data_[7] - data_[6] * data_[4];
        inv(2,1) = data_[1] * data_[6] - data_[0] * data_[7];
        inv(2,2) = data_[0] * data_[4] - data_[1] * data_[3];

        inv *= (1 / det);
        return inv;
    }

    AxMatrix3x3<T>& operator = (const AxMatrix3x3<T>& other)
    {
        memcpy(data_, other.data(), sizeof(T) * 9);
        return *this;
    }

    AxMatrix3x3<T>& operator +=(const AxMatrix3x3<T>& other)
    {
        AX_FOR_I(9)
        {
            data_[i] += other.data()[i];
        }
        return *this;
    }

    AxMatrix3x3<T>& operator -=(const AxMatrix3x3<T>& other)
    {
        AX_FOR_I(9)
        {
            data_[i] -= other.data()[i];
        }
        return *this;
    }

    AxMatrix3x3<T>& operator *=(T scalar)
    {
        AX_FOR_I(9)
        {
            data_[i] *= scalar;
        }
        return *this;
    }
};

#endif