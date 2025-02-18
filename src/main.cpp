#include <iostream>
#include "Core/Math/Quaternion.h"
#include "Core/Math/Vector3.h"


using namespace PhysicsEngine::Math;

int main() {
    // 构造一个四元数，表示绕 (0, 1, 0) 轴旋转 90 度（1.5708 弧度）
    Vector3 axis(0.0f, 1.0f, 0.0f);
    Scalar angle = 1.5708f; // 90度，单位：弧度
    Quaternion q = Quaternion::fromAxisAngle(axis, angle);
    std::cout << "Quaternion from axis-angle: " << q << std::endl;

    // 验证共轭与求逆
    Quaternion qConj = q.Conjugate();
    Quaternion qInv = q.Inverse();
    std::cout << "Conjugate: " << qConj << std::endl;
    std::cout << "Inverse: " << qInv << std::endl;

    // 四元数与其逆相乘应当得到单位四元数
    Quaternion identity = q * qInv;
    std::cout << "q * qInv = " << identity << std::endl;

    return 0;
}
