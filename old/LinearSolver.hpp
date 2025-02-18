#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

#include "LinearAlgebra.hpp"


template<typename dataType>
void LUDecomposition(const dataType* A, dataType* LU, AxUInt32* pivot, AxUInt32 m)
{
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < m; ++j) // for all cols
        {
            
        }
    }
}

template<typename dataType>
void LUBackSubstitute(const dataType* LU, const AxUInt32* pivot, const dataType* b, dataType* r, AxUInt32 m)
{
    
}

#endif

