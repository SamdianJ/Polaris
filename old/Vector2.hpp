#ifndef AXVECTOR_H
#define AXVECTOR_H

#include "dataType.h"

namespace Polaris
{
    template <typename T>
    struct AxVector2
    {
        T x;
        T y;

        T& operator[] (const AxUInt32 i)
        {
            if (i > 2)
            {
                std::cout << "inValid operator Id" << std::endl;
                exit(-1)
            }
            return i == 0 ? x : y;
        }
    };

    template <typename T>
    struct AxVector3
    {
        T x;
        T y;
        T z;

        T& operator[] (const AxUInt32 i)
        {
            if (i > 3)
            {
                std::cout << "inValid operator Id" << std::endl;
                exit(-1)
            }
            return i == 0 ? x : (i == 1 ? y : z) ;
        }
    };
}

#endif
