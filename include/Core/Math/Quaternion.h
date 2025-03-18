#ifndef PHYSICSENGINE_MATH_QUATERNION_H
#define PHYSICSENGINE_MATH_QUATERNION_H

#include <cmath>
#include <iostream>
#include "Vector3.h"  // 包含 Vector3 类，用于 fromAxisAngle
#include "Matrix3.h"

namespace Polaris {
namespace Math {

class Quaternion {
public:
    // 四元数的分量：w 为标量部分，(x, y, z) 为向量部分
    Scalar w, x, y, z;

    // 默认构造函数，生成单位四元数
    HOST_DEVICE_FUNC Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}

    // 构造函数：直接指定四元数的分量
    HOST_DEVICE_FUNC Quaternion(Scalar w, Scalar x, Scalar y, Scalar z)
        : w(w), x(x), y(y), z(z) {}

    // 四元数加法：返回两个四元数对应分量相加后的新四元数
    HOST_DEVICE_FUNC Quaternion operator+(const Quaternion& other) const {
        return Quaternion(w + other.w, x + other.x, y + other.y, z + other.z);
    }

    // 四元数减法：返回两个四元数对应分量相减后的新四元数
    HOST_DEVICE_FUNC Quaternion operator-(const Quaternion& other) const {
        return Quaternion(w - other.w, x - other.x, y - other.y, z - other.z);
    }

    // 复合赋值加法运算符
    HOST_DEVICE_FUNC Quaternion& operator+=(const Quaternion& other) {
        w += other.w;
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    // 复合赋值减法运算符
    HOST_DEVICE_FUNC Quaternion& operator-=(const Quaternion& other) {
        w -= other.w;
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    // 四元数乘法：将当前四元数与另一个四元数相乘，注意乘法不满足交换律
    HOST_DEVICE_FUNC Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,  // 新的标量部分
            w * other.x + x * other.w + y * other.z - z * other.y,  // 新的 x 分量
            w * other.y - x * other.z + y * other.w + z * other.x,  // 新的 y 分量
            w * other.z + x * other.y - y * other.x + z * other.w   // 新的 z 分量
        );
    }

    // 复合赋值乘法运算符
    HOST_DEVICE_FUNC Quaternion& operator*=(const Quaternion& other) {
        *this = (*this) * other;
        return *this;
    }

    // 定义和标量的乘法（Quaternion * scalar）
    HOST_DEVICE_FUNC Quaternion operator*(Scalar scalar) const {
        return Quaternion(w * scalar, x * scalar, y * scalar, z * scalar);
    }

    // 复合赋值标量乘法运算符（Quaternion *= scalar）
    HOST_DEVICE_FUNC Quaternion& operator*=(Scalar scalar) {
        w *= scalar;
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    // 计算四元数的模长
    HOST_DEVICE_FUNC Scalar Norm() const {
        return std::sqrt(w * w + x * x + y * y + z * z);
    }


    // 归一化当前四元数
    HOST_DEVICE_FUNC Quaternion Normalize() const {
        Scalar n = Norm();
        if (n > 0)
            return Quaternion(w / n, x / n, y / n, z / n);
        // 若模长为 0，则返回单位四元数
        return Quaternion();
    }

    // 返回归一化后的四元数
    HOST_DEVICE_FUNC Quaternion Normalized() const {
        Scalar n = Norm();
        if (n > 0)
            return Quaternion(w / n, x / n, y / n, z / n);
        // 若模长为 0，则返回单位四元数
        return Quaternion();
    }

    // 返回四元数的共轭
    HOST_DEVICE_FUNC Quaternion Conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }

    // 返回四元数的逆：逆等于共轭除以模长的平方
    HOST_DEVICE_FUNC Quaternion Inverse() const {
        Scalar n = Norm();
        if (n > 0) {
            Scalar invNormSq = 1.0f / (n * n);
            Quaternion conj = Conjugate();
            return Quaternion(conj.w * invNormSq, conj.x * invNormSq, conj.y * invNormSq, conj.z * invNormSq);
        }
        // 若模长为 0，则无法求逆，返回单位四元数
        return Quaternion();
    }

    HOST_DEVICE_FUNC Math::Matrix3 ToRotationMatrix() const {
        // 若未归一化，可先归一化当前四元数
        Quaternion q = Normalized();

        Scalar xx = q.x * q.x;
        Scalar yy = q.y * q.y;
        Scalar zz = q.z * q.z;
        Scalar xy = q.x * q.y;
        Scalar xz = q.x * q.z;
        Scalar yz = q.y * q.z;
        Scalar wx = q.w * q.x;
        Scalar wy = q.w * q.y;
        Scalar wz = q.w * q.z;

        return Matrix3(
            1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy),
            2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
            2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy)
        );
    }

    // 根据轴角构造四元数，axis 为旋转轴（应为归一化向量），angle 为旋转角度（单位：弧度）
    HOST_DEVICE_FUNC static Quaternion fromAxisAngle(const Vector3& axis, Scalar angle) {
        Scalar halfAngle = angle * 0.5f;
        Scalar sinHalf = std::sin(halfAngle);
        Scalar cosHalf = std::cos(halfAngle);
        // 生成的四元数为：cos(halfAngle) + sin(halfAngle) * (axis.x*i + axis.y*j + axis.z*k)
        return Quaternion(cosHalf, axis.x * sinHalf, axis.y * sinHalf, axis.z * sinHalf).Normalized();
    }

    HOST_DEVICE_FUNC static Quaternion Identity() {
        return Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
    }
};

// 定义标量在左侧乘以四元数的运算符（scalar * Quaternion）
HOST_DEVICE_FUNC inline Quaternion operator*(Scalar scalar, const Quaternion& q) {
    return q * scalar;
}

// 重载输出运算符，便于调试时输出四元数信息
HOST_FUNC inline std::ostream& operator<<(std::ostream& os, const Quaternion& q) {
    os << "(" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")";
    return os;
}

} // namespace Math
} // namespace PhysicsEngine

#endif // PHYSICSENGINE_MATH_QUATERNION_H
