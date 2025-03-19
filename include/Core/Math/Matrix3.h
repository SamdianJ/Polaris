#ifndef PHYSICSENGINE_MATH_MATRIX3_H
#define PHYSICSENGINE_MATH_MATRIX3_H

#include <cmath>
#include <iostream>
#include "Polaris.h"

namespace Polaris {
namespace Math {

class Matrix3 {
public:
    // 以行优先顺序存储矩阵元素，共 9 个浮点数
    Scalar m[9];

    // 默认构造函数，生成单位矩阵
    HOST_DEVICE_FUNC Matrix3() {
        m[0] = 1.0f; m[1] = 0.0f; m[2] = 0.0f;
        m[3] = 0.0f; m[4] = 1.0f; m[5] = 0.0f;
        m[6] = 0.0f; m[7] = 0.0f; m[8] = 1.0f;
    }

    // 构造函数，按行优先顺序指定9个元素
    HOST_DEVICE_FUNC Matrix3(Scalar m0, Scalar m1, Scalar m2,
        Scalar m3, Scalar m4, Scalar m5,
        Scalar m6, Scalar m7, Scalar m8) {
        m[0] = m0; m[1] = m1; m[2] = m2;
        m[3] = m3; m[4] = m4; m[5] = m5;
        m[6] = m6; m[7] = m7; m[8] = m8;
    }

    // 下标运算符，获取第 i 行第 j 列的元素（0 <= i,j < 3）
    HOST_DEVICE_FUNC Scalar& operator()(int i, int j) {
        return m[i * 3 + j];
    }
    
    HOST_DEVICE_FUNC const Scalar& operator()(int i, int j) const {
        return m[i * 3 + j];
    }

    // 矩阵加法
    HOST_DEVICE_FUNC Matrix3 operator+(const Matrix3 &other) const {
        return Matrix3(
            m[0] + other.m[0], m[1] + other.m[1], m[2] + other.m[2],
            m[3] + other.m[3], m[4] + other.m[4], m[5] + other.m[5],
            m[6] + other.m[6], m[7] + other.m[7], m[8] + other.m[8]
        );
    }

    // 矩阵减法
    HOST_DEVICE_FUNC Matrix3 operator-(const Matrix3 &other) const {
        return Matrix3(
            m[0] - other.m[0], m[1] - other.m[1], m[2] - other.m[2],
            m[3] - other.m[3], m[4] - other.m[4], m[5] - other.m[5],
            m[6] - other.m[6], m[7] - other.m[7], m[8] - other.m[8]
        );
    }

    // 标量乘法
    HOST_DEVICE_FUNC Matrix3 operator*(Scalar scalar) const {
        return Matrix3(
            m[0] * scalar, m[1] * scalar, m[2] * scalar,
            m[3] * scalar, m[4] * scalar, m[5] * scalar,
            m[6] * scalar, m[7] * scalar, m[8] * scalar
        );
    }

    // 矩阵乘法（矩阵相乘）
    HOST_DEVICE_FUNC Matrix3 operator*(const Matrix3 &other) const {
        Matrix3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Scalar sum = 0.0f;
                for (int k = 0; k < 3; k++) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // 计算行列式
    HOST_DEVICE_FUNC Scalar Determinant() const {
        return m[0]*(m[4]*m[8] - m[5]*m[7]) -
               m[1]*(m[3]*m[8] - m[5]*m[6]) +
               m[2]*(m[3]*m[7] - m[4]*m[6]);
    }

    // 计算转置矩阵
    HOST_DEVICE_FUNC Matrix3 Transpose() const {
        return Matrix3(
            m[0], m[3], m[6],
            m[1], m[4], m[7],
            m[2], m[5], m[8]
        );
    }

    // 计算逆矩阵，如果行列式为 0，则返回单位矩阵（可根据需要修改错误处理策略）
    HOST_DEVICE_FUNC Matrix3 Inverse() const {
        Scalar det = Determinant();
        if (fabs(det) < 1e-6) { // 行列式近似为0
            return Matrix3(); // 返回单位矩阵
        }
        Scalar invDet = 1.0f / det;
        Matrix3 inv;
        inv(0,0) =  (m[4]*m[8] - m[5]*m[7]) * invDet;
        inv(0,1) = -(m[1]*m[8] - m[2]*m[7]) * invDet;
        inv(0,2) =  (m[1]*m[5] - m[2]*m[4]) * invDet;

        inv(1,0) = -(m[3]*m[8] - m[5]*m[6]) * invDet;
        inv(1,1) =  (m[0]*m[8] - m[2]*m[6]) * invDet;
        inv(1,2) = -(m[0]*m[5] - m[2]*m[3]) * invDet;

        inv(2,0) =  (m[3]*m[7] - m[4]*m[6]) * invDet;
        inv(2,1) = -(m[0]*m[7] - m[1]*m[6]) * invDet;
        inv(2,2) =  (m[0]*m[4] - m[1]*m[3]) * invDet;
        return inv;
    }

    HOST_DEVICE_FUNC static Matrix3 Identity()
    {
        return Matrix3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    }
};

// 重载标量乘法运算符（标量在左侧）
HOST_DEVICE_FUNC inline Matrix3 operator*(Scalar scalar, const Matrix3 &mat) {
    return mat * scalar;
}

// 定义 Matrix3 与 Vector3 的乘法，计算 M * v 的结果（v为列向量）
HOST_DEVICE_FUNC inline Vector3 operator*(const Matrix3& mat, const Vector3& vec) {
    return Vector3(
        mat(0, 0) * vec.x + mat(0, 1) * vec.y + mat(0, 2) * vec.z,
        mat(1, 0) * vec.x + mat(1, 1) * vec.y + mat(1, 2) * vec.z,
        mat(2, 0) * vec.x + mat(2, 1) * vec.y + mat(2, 2) * vec.z
    );
}

// 重载输出运算符，便于打印矩阵
HOST_FUNC inline std::ostream& operator<<(std::ostream &os, const Matrix3 &mat) {
    os << "[" << mat(0,0) << ", " << mat(0,1) << ", " << mat(0,2) << "]\n"
       << "[" << mat(1,0) << ", " << mat(1,1) << ", " << mat(1,2) << "]\n"
       << "[" << mat(2,0) << ", " << mat(2,1) << ", " << mat(2,2) << "]";
    return os;
}

} // namespace Math
} // namespace PhysicsEngine

#endif // PHYSICSENGINE_MATH_MATRIX3_H
