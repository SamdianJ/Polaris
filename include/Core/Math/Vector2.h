#ifndef PHYSICSENGINE_MATH_VECTOR2_H
#define PHYSICSENGINE_MATH_VECTOR2_H

#include <cmath>
#include "PolarisMacro.h"
#include <iostream>

namespace Polaris {
namespace Math {

class Vector2 {
public:
    Scalar x, y;

    // 构造函数
    HOST_DEVICE_FUNC Vector2() : x(0.0f), y(0.0f) {}
    HOST_DEVICE_FUNC Vector2(Scalar x, Scalar y) : x(x), y(y) {}

    // 运算符重载：加法
    HOST_DEVICE_FUNC Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }
    
    // 运算符重载：减法
    HOST_DEVICE_FUNC Vector2 operator-(const Vector2& other) const {
        return Vector2(x - other.x, y - other.y);
    }
    
    // 运算符重载：标量乘法
    HOST_DEVICE_FUNC Vector2 operator*(Scalar scalar) const {
        return Vector2(x * scalar, y * scalar);
    }
    
    // 运算符重载：标量除法
    HOST_DEVICE_FUNC Vector2 operator/(Scalar scalar) const {
        return Vector2(x / scalar, y / scalar);
    }
    
    // 复合赋值运算符
    HOST_DEVICE_FUNC Vector2& operator+=(const Vector2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    
    HOST_DEVICE_FUNC Vector2& operator-=(const Vector2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    
    HOST_DEVICE_FUNC Vector2& operator*=(Scalar scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    
    HOST_DEVICE_FUNC Vector2& operator/=(Scalar scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }
    
    // 计算向量长度
    HOST_DEVICE_FUNC Scalar length() const {
        return std::sqrt(x * x + y * y);
    }
    
    // 计算向量的平方长度（避免不必要的开方运算）
    HOST_DEVICE_FUNC Scalar squaredLength() const {
        return x * x + y * y;
    }
    
    // 返回归一化后的向量
    HOST_DEVICE_FUNC Vector2 normalized() const {
        Scalar len = length();
        if (len > 0)
            return Vector2(x / len, y / len);
        return Vector2(0.0f, 0.0f);
    }
    
    // 点乘运算
    HOST_DEVICE_FUNC Scalar dot(const Vector2& other) const {
        return x * other.x + y * other.y;
    }
};

// 重载输出运算符，便于调试
HOST_FUNC inline std::ostream& operator<<(std::ostream& os, const Vector2& vec) {
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}

} // namespace Math
} // namespace PhysicsEngine

#endif // PHYSICSENGINE_MATH_VECTOR2_H
