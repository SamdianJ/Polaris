#ifndef AXMATRIX_H
#define AXMATRIX_H

#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include "polaris.h"

namespace Polaris
{
    namespace LinearAlgebra
    {
        // template <typename dataType>
        // class AxDenseMatrixMN
        // {

        // private:
        //     std::vector<dataType> data;
        //     std::string m_Name;
        //     AxUInt32 nRow_;
        //     AxUInt32 nCol_;

        // public:
        //     AxDenseMatrixMN()
        //     :
        //     data(std::vector<dataType> {0}),
        //     nRow_(0),
        //     nCol_(0)
        //     {}

        //     AxDenseMatrixMN(AxUInt32 m, AxUInt32 n)
        //     :
        //     nRow_(m),
        //     nCol_(n)
        //     {
        //         rawData_ = malloc()
        //     }

        //     AxUInt32 m()
        //     {
        //         return nRow_;
        //     }

        //     AxUInt32 n()
        //     {
        //         return nCol_;
        //     }

        //     AxUInt32 size()
        //     {
        //         return nRow_ * nCol_;
        //     }

        //     AxUInt32 resize(Axm1, n1)
        //     {
        //         allocator.reallocate(m1 * n1);
        //         nRow_ = m1;
        //         nCol_ = n1;
        //     }

        //     AxUInt32 shallowResize(m1, n1)
        //     {
        //         nRow_ = m1;
        //         nCol_ = n1;
        //     }

        //     // visit
        //     const dataType &operator()(const AxUInt32 x, const AxUInt32 y) const
        //     {
        //         return rawData_[x * nRow_ + y];
        //     }

        //     dataType &operator()(const AxUInt32 x, const AxUInt32 y)
        //     {
        //         return rawData_[x * nRow_ + y];
        //     }

        //     const dataType &operator[](const AxUInt32 x) const
        //     {
        //         return rawData_[x];
        //     }

        //     dataType &operator[](const AxUInt32 x)
        //     {
        //         return rawData_[x];
        //     }

        //     dataType* RawData()
        //     {
        //         return rawData_;
        //     }

        //     void Print() const
        //     {
        //         std::cout << m_Name << "Dense-MatrixMN " << "of size: (" << m() << " , " << n() << ") " << std::endl;
        //         std::cout << "Data: [" << std::endl;
        //         AX_FOR_I(m())
        //         {
        //             AX_FOR_J(n() - 1)
        //             {
        //                 std::cout << (*this)(i, j) << ", ";
        //             }
        //             std::cout << (*this)(i, n() - 1) << std::endl;
        //         }
        //         std::cout << "]" << std::endl;
        //     }

        // };

        template <typename dataType, AxUInt32 mDim, AxUInt32 nDim>
        class AxDenseMatrix
        {
        private:
            std::string m_Name;
            std::vector<dataType> m_Data;

        public:
            explicit AxDenseMatrix(std::string name)
                : 
                m_Name(name),
                m_Data(mDim * nDim)
            {}

            explicit AxDenseMatrix()
                : 
                m_Name("UnNamed-Dense-Matrix"),
                m_Data(mDim * nDim)
            {}

            virtual ~AxDenseMatrix()
            {}

            const std::string Name()
            {
                return m_Name;
            }

            // visit
            const dataType &operator()(const AxUInt32 x, const AxUInt32 y) const
            {
                return m_Data[x * nDim + y];
            }

            dataType &operator()(const AxUInt32 x, const AxUInt32 y)
            {
                return m_Data[x * nDim + y];
            }

            const dataType &operator[](const AxUInt32 x) const
            {
                return m_Data[x];
            }

            dataType &operator[](const AxUInt32 x)
            {
                return m_Data[x];
            }

            dataType* RawData()
            {
                return m_Data.data();
            }

            void Print() const
            {
                std::cout << m_Name << "Dense-Matrix " << "of size: (" << mDim << " , " << nDim << ") " << std::endl;
                std::cout << "Data: [" << std::endl;
                AX_FOR_I(mDim)
                {
                    AX_FOR_J(nDim - 1)
                    {
                        std::cout << (*this)(i, j) << ", ";
                    }
                    std::cout << (*this)(i, nDim - 1) << std::endl;
                }
                std::cout << "]" << std::endl;
            }
        };

        template <typename dataType, AxUInt32 dim>
        using AxDenseArray = AxDenseMatrix<dataType, dim, 1>;

        template <typename dataType, AxUInt32 mDim, AxUInt32 nDim>
        class AxSparseCSRMatrix
        {
        private:
            std::string m_Name;
            AxUInt32 m_NumNonZero;
            AxFp32 m_Sparsity;
            std::vector<dataType> m_Data;
            std::vector<AxUInt32> m_ColumeIndex;
            std::vector<AxUInt32> m_RowOffset;

            void zip()
            {
                m_Data.resize(m_NumNonZero);
                m_ColumeIndex.resize(m_NumNonZero);
                m_Sparsity = (AxFp32) m_NumNonZero / (AxFp32) (mDim * nDim);
            }

            void zip(AxUInt32 nnz)
            {
                m_NumNonZero = nnz;
                m_Data.resize(m_NumNonZero);
                m_ColumeIndex.resize(m_NumNonZero);
                m_Sparsity = (AxFp32) m_NumNonZero / (AxFp32) (mDim * nDim);
            }

            AxUInt32 convertFromDense(const dataType* source, dataType* data, AxUInt32* columeIndex, AxUInt32* rowOffset, AxFp32 threshold)
            {
                AxUInt32 numNonZero = 0;
                rowOffset[0] = 0;
                for (int i = 0; i < mDim; ++i)
                {
                    AxUInt32 numRowNonZero = 0;
                    for (int j = 0; j < nDim; ++j)
                    {
                        if (PLS_ABS(source[i * mDim + j] > threshold))
                        {
                            data[numNonZero] = source[i * mDim + j];
                            columeIndex[numNonZero] = j;
                            numNonZero++;
                            numRowNonZero++;
                        }
                    }
                    rowOffset[i + 1] = numNonZero;
                }

                return numNonZero;
            }

        public:
            explicit AxSparseCSRMatrix(std::string name)
                : 
                m_Name(name),
                m_NumNonZero(0),
                m_Sparsity(0.0f),
                m_Data(),
                m_ColumeIndex(),
                m_RowOffset(mDim + 1, 0)
            {}

            explicit AxSparseCSRMatrix(AxDenseMatrix<dataType, mDim, nDim>& DenseMatrix)
                : 
                m_Name(DenseMatrix.Name()),
                m_NumNonZero(0),
                m_Sparsity(0.0f),
                m_Data(mDim * nDim),
                m_ColumeIndex(mDim * nDim),
                m_RowOffset(mDim + 1, 0)
            {
                m_NumNonZero = convertFromDense(DenseMatrix.RawData(), m_Data.data(), m_ColumeIndex.data(), m_RowOffset.data(), 1e-6f);
                m_Sparsity = (AxFp32)m_NumNonZero / (AxFp32)(mDim * nDim);
            }

            explicit AxSparseCSRMatrix()
                : 
                m_Name("UnNamed-Sparse-Matrix"),
                m_NumNonZero(0),
                m_Sparsity(0.0f),
                m_Data(),
                m_ColumeIndex(),
                m_RowOffset(mDim + 1, 0)
            {}

            virtual ~AxSparseCSRMatrix()
            {}

            std::string Name() const
            {
                return m_Name;
            }

            dataType* Data()
            {
                return m_Data.data();
            }

            AxUInt32* columeIndex()
            {
                return m_ColumeIndex.data();
            }

            AxUInt32* rowOffset()
            {
                return m_RowOffset.data();
            }

            AxUInt32 numNonZero() const
            {
                return m_NumNonZero;
            }

            AxFp32 sparsity() const
            {
                return m_Sparsity;
            }

            void correct()
            {
                zip(); 
            }

            void correct(AxUInt32 nnz)
            {
                zip(nnz); 
            }

            void reserve(AxUInt32 nnz)
            {
                m_Data.resize(nnz);
                m_ColumeIndex.resize(nnz);
            }

            void Print() const
            {                  
                std::cout << m_Name << "Sparse-Matrix " << "of size: (" << mDim << " , " << nDim << ") " << std::endl;
                std::cout << "Number of non-zero elements: " << m_NumNonZero << ", with sparsity: " << m_Sparsity << std::endl;
                std::cout << "data: [";
                AX_FOR_I(m_NumNonZero)
                {
                    std::cout << m_Data[i] << " ";
                }
                std::cout << "]" << std::endl;
                std::cout << "colume index: [";
                AX_FOR_I(m_NumNonZero)
                {
                    std::cout << m_ColumeIndex[i] << " ";
                }
                std::cout << "]" << std::endl;
                std::cout << "row offset: [";
                AX_FOR_I(mDim + 1)
                {
                    std::cout << m_RowOffset[i] << " ";
                }
                std::cout << "]" << std::endl;
            }

            AxDenseMatrix<dataType, mDim, nDim> convertToDense()
            {
                AxDenseMatrix<dataType, mDim, nDim> matrix;

                AX_FOR_I(mDim)
                {
                    AxUInt32 offset = m_RowOffset[i];
                    AxUInt32 numRowNonZero = m_RowOffset[i + 1] - offset;

                    AX_FOR_J(numRowNonZero)
                    {
                        matrix(i, m_ColumeIndex[offset + j]) = m_Data[offset + j];
                    }
                }

                return matrix;
            }
        };

        template <typename dataType>
        void gemm(const dataType* A, const dataType* B, dataType* C, AxUInt32 m1, AxUInt32 n1, AxUInt32 m2, AxUInt32 n2)
        {
            for (int i = 0; i < m1; ++i)
            {
                for (int j = 0; j < n2; ++j)
                {
                    dataType tmp = (dataType)0.0;
                    for (int k = 0; k < n1; ++k)
                    {
                        tmp += A[i * m1 + k] * B[k * n2 + j];
                    }

                    C[i * m1 + j] = tmp;
                }
            }
        }

        template <typename dataType>
        void gemv(const dataType* A, const dataType* v, dataType* r, AxUInt32 m, AxUInt32 n)
        {
            for (int i = 0; i < m; ++i)
            {
                r[i] = (dataType) 0;
                for (int j = 0; j < n; ++j)
                {
                    r[i] += A[m * i + j] * v[j]; 
                }
            }
        }

        template <typename dataType>
        void semm(const dataType* A_data, const AxUInt32* A_columeIndex, const AxUInt32* A_rowOffset, 
                  const dataType* B_data, const AxUInt32* B_columeIndex, const AxUInt32* B_rowOffset, 
                  dataType* tmp, dataType* C_data, AxUInt32* C_columeIndex, AxUInt32* C_rowOffset, 
                  AxUInt32 m1, AxUInt32 n1, AxUInt32 m2, AxUInt32 n2, AxFp32 threshold, AxUInt32& nnz)
        {
            C_rowOffset[0] = 0;
            AxUInt32 numResultNonZero = 0;
            for (int i = 0; i < m1; ++i)
            {
                AxUInt32 offset = A_rowOffset[i];
                AxUInt32 numRowElements = A_rowOffset[i + 1] - offset;

                AxUInt32 numResultRowNonZero = 0;
                memset(tmp, 0, sizeof(dataType) * (n2));
                for (int j = 0; j < numRowElements; ++j)
                {
                    AxUInt32 colIndex = A_columeIndex[offset + j]; // get the number of row index to perform
                    dataType data = A_data[offset + j];

                    AxUInt32 numRowElements_b = B_rowOffset[colIndex + 1] - B_rowOffset[colIndex];
                    for (int k = 0; k < numRowElements_b; ++k)
                    {
                        AxUInt32 columeIndex_b = B_columeIndex[B_rowOffset[colIndex] + k];
                        tmp[columeIndex_b] += data * B_data[B_rowOffset[colIndex] + k];
                    }
                }

                for (int j = 0; j < n2; ++j)
                {
                    if (PLS_ABS(tmp[j]) > threshold)
                    {
                        C_data[numResultNonZero] = tmp[j];
                        C_columeIndex[numResultNonZero] = j;
                        numResultRowNonZero++;
                        numResultNonZero++;
                    }
                }
                C_rowOffset[i + 1] = numResultNonZero;
            }

            nnz = numResultNonZero;
        }

        template <typename dataType>
        void semv(const dataType* A_data, const AxUInt32* A_columeIndex, const AxUInt32* A_rowOffset, 
                  const dataType* v, dataType* r, AxUInt32 m, AxUInt32 n)
        {
            for (int i = 0; i < m; ++i)
            {
                AxUInt32 rowOffset = A_rowOffset[i];
                AxUInt32 numRowNonZero = A_rowOffset[i + 1] - rowOffset;
                r[i] = (dataType) 0;
                for (int j = 0; j < numRowNonZero; ++j)
                {
                    r[i] += A_data[rowOffset + j] * v[A_columeIndex[rowOffset + j]];
                }
            }
        }

        template <typename dataType, AxUInt32 m_Dim, AxUInt32 n_Dim>
        AxSparseCSRMatrix<dataType, m_Dim, n_Dim> sparseFromDense(const AxDenseMatrix<dataType, m_Dim, n_Dim>& dense)
        {
            AxSparseCSRMatrix<dataType, m_Dim, n_Dim> sparse(dense);
            return sparse;
        }

        template <typename dataType, AxUInt32 m1_Dim, AxUInt32 n1_Dim, AxUInt32 m2_Dim, AxUInt32 n2_Dim>
        void DenseMatMatMult(AxDenseMatrix<dataType, m1_Dim, n1_Dim>& A, AxDenseMatrix<dataType, m2_Dim, n2_Dim>& B, AxDenseMatrix<dataType, m1_Dim, n2_Dim>& C, AxUInt32 platform)
        {
            static_assert(
                n1_Dim == m2_Dim,
                "Dense matrix-matrix multiplication must take compatible dimensions"
                );

            switch (platform)
            {
            case 1: //CPU-x86
                gemm(A.RawData(), B.RawData(), C.RawData(), m1_Dim, n1_Dim, m2_Dim, n2_Dim);
                break;
            default:
                gemm(A.RawData(), B.RawData(), C.RawData(), m1_Dim, n1_Dim, m2_Dim, n2_Dim);
                break;
            }
        }

        template <typename dataType, AxUInt32 m1_Dim, AxUInt32 n1_Dim, AxUInt32 m2_Dim>
        void DenseMatVecMult(AxDenseMatrix<dataType, m1_Dim, n1_Dim>& A, AxDenseMatrix<dataType, m2_Dim, 1>& v, AxDenseMatrix<dataType, m1_Dim, 1>& r, AxUInt32 platform)
        {
            static_assert(
                n1_Dim == m2_Dim,
                "Dense matrix-vector multiplication must take compatible dimensions"
                );

            switch (platform)
            {
            case 1: //CPU-x86
                gemv(A.RawData(), v.RawData(), r.RawData(), m1_Dim, n1_Dim);
                break;
            default:
                gemv(A.RawData(), v.RawData(), r.RawData(), m1_Dim, n1_Dim);
                break;
            }
        }

        template <typename dataType, AxUInt32 m1_Dim, AxUInt32 n1_Dim, AxUInt32 m2_Dim, AxUInt32 n2_Dim>
        void SparseMatMatMult(AxSparseCSRMatrix<dataType, m1_Dim, n1_Dim>& A, AxSparseCSRMatrix<dataType, m2_Dim, n2_Dim>& B, AxSparseCSRMatrix<dataType, m1_Dim, n2_Dim>& C, AxUInt32 platform)
        {
            static_assert(
                n1_Dim == m2_Dim,
                "Sparse matrix-vector multiplication must take compatible dimensions"
                );

            C.reserve(m1_Dim * n2_Dim);
            std::vector<dataType> tmp(n2_Dim, 0);
            AxUInt32 nnz = 0;

            switch (platform)
            {
            case 1: //CPU-x86            
                semm(A.Data(), A.columeIndex(), A.rowOffset(), B.Data(), B.columeIndex(), B.rowOffset(),
                tmp.data(), C.Data(), C.columeIndex(), C.rowOffset(), m1_Dim, n1_Dim, m2_Dim, n2_Dim, 1e-6f, nnz);
                break;
            default:
                semm(A.Data(), A.columeIndex(), A.rowOffset(), B.Data(), B.columeIndex(), B.rowOffset(),
                tmp.data(), C.Data(), C.columeIndex(), C.rowOffset(), m1_Dim, n1_Dim, m2_Dim, n2_Dim, 1e-6f, nnz);
                break;
            }
            C.correct(nnz);
        }

        template <typename dataType, AxUInt32 m1_Dim, AxUInt32 n1_Dim, AxUInt32 m2_Dim>
        void SparseMatVecMult(AxSparseCSRMatrix<dataType, m1_Dim, n1_Dim>& A, AxDenseArray<dataType, m2_Dim>& v, AxDenseArray<dataType, m2_Dim>& r, AxUInt32 platform)
        {
            static_assert(
                n1_Dim == m2_Dim,
                "Sparse matrix-vector multiplication must take compatible dimensions"
                );

            switch (platform)
            {
            case 1: //CPU-x86            
                semv(A.Data(), A.columeIndex(), A.rowOffset(), v.RawData(), r.RawData(), m1_Dim, n1_Dim);
                break;
            default:
                semv(A.Data(), A.columeIndex(), A.rowOffset(), v.RawData(), r.RawData(), m1_Dim, n1_Dim);
                break;
            }
        }
    }
}

#endif