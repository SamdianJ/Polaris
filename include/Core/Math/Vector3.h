#ifndef PHYSICSENGINE_MATH_VECTOR3_H
#define PHYSICSENGINE_MATH_VECTOR3_H

#include <cmath>
#include <iostream>
#include "PolarisMacro.h"

namespace PhysicsEngine {
namespace Math {

class Vector3 {
public:
    Scalar x, y, z;

    // 构造函数
    Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3(Scalar x, Scalar y, Scalar z) : x(x), y(y), z(z) {}

    // 运算符重载：加法
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    
    // 运算符重载：减法
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    
    // 运算符重载：标量乘法
    Vector3 operator*(Scalar scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }
    
    // 运算符重载：标量除法
    Vector3 operator/(Scalar scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }
    
    // 复合赋值运算符
    Vector3& operator+=(const Vector3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    
    Vector3& operator-=(const Vector3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    
    Vector3& operator*=(Scalar scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    
    Vector3& operator/=(Scalar scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
    
    // 计算向量长度
    Scalar Length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    // 计算向量的平方长度
    Scalar SquaredLength() const {
        return x * x + y * y + z * z;
    }
    
    // 返回归一化后的向量
    Vector3 Normalized() const {
        Scalar len = Length();
        if (len > 0)
            return Vector3(x / len, y / len, z / len);
        return Vector3(0.0f, 0.0f, 0.0f);
    }
    
    // 点乘运算
    Scalar Dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    // 叉乘运算
    Vector3 Cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    static Vector3 Zero()
    {
        return Vector3(0.0f, 0.0f, 0.0f);
    }

    static Vector3 Cross(const Vector3& a, const Vector3& b)
    {
        return Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
};

// 重载输出运算符，便于调试
inline std::ostream& operator<<(std::ostream& os, const Vector3& vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

} // namespace Math
} // namespace PhysicsEngine

#endif // PHYSICSENGINE_MATH_VECTOR3_H
