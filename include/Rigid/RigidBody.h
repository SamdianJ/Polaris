#pragma once

#include "Core/Math/Vector3.h"
#include "Core/Math/Quaternion.h"
#include "Core/Math/Matrix3.h"

namespace PhysicsEngine
{
    class RigidBody
    {
    public:
        // 刚体属性
        Scalar mass;                // 质量
        Scalar inverseMass;         // 质量的倒数（优化计算）
        Math::Vector3 position;          // 位置
        Math::Vector3 velocity;          // 线速度
        Math::Vector3 forceAccum;        // 累积的力
        Math::Quaternion orientation;    // 旋转（四元数）
        Math::Vector3 angularVelocity;   // 角速度
        Math::Vector3 torqueAccum;       // 累积的扭矩
        Math::Matrix3 inverseInertia;    // 惯性张量的逆（局部坐标系）
        Math::Matrix3 inverseInertiaWorld; // 惯性张量的逆（世界坐标系）

    public:
        RigidBody();

        // 设置质量（自动计算 inverseMass）
        void SetMass(Scalar mass);

        // 应用力到质心
        void ApplyForce(const Math::Vector3& force);

        // 应用力到指定点（生成力和扭矩）
        void ApplyForceAtPoint(const Math::Vector3& force, const Math::Vector3& point);

        // 应用扭矩
        void ApplyTorque(const Math::Vector3& torque);

        // 更新刚体状态（积分）
        void Update(Scalar deltaTime);

        // 清空累积的力和扭矩
        void ClearAccumulators();

    private:
        // 更新惯性张量到世界坐标系
        void UpdateInertiaTensor();
    };
}