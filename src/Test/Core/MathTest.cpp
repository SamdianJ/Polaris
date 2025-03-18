#include <iostream>
#include "Core/Math/Quaternion.h"
#include "Core/Math/Vector3.h"
#include "Test/Core/MathTest.h"

using namespace Polaris;

PlsInt32 Polaris::MathTest()
{
    Math::Vector3 axis(0.0f, 1.0f, 0.0f);
    Scalar angle = 1.5708f;
    Math::Quaternion q = Math::Quaternion::fromAxisAngle(axis, angle);
    std::cout << "Quaternion from axis-angle: " << q << std::endl;

    Math::Quaternion qConj = q.Conjugate();
    Math::Quaternion qInv = q.Inverse();
    std::cout << "Conjugate: " << qConj << std::endl;
    std::cout << "Inverse: " << qInv << std::endl;

    Math::Quaternion identity = q * qInv;
    std::cout << "q * qInv = " << identity << std::endl;

    if (abs(identity.Norm() - 1.0) < 1e-6)
    {
        std::cout << "Math test passes" << std::endl;
        return 0;
    }
    return -1;
}

