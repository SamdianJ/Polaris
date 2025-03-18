#include "Rigid/RigidBody.h"

namespace Polaris
{
    RigidBody::RigidBody()
        : mass(1.0f), inverseMass(1.0f),
          position(Math::Vector3::Zero()), velocity(Math::Vector3::Zero()),
          forceAccum(Math::Vector3::Zero()), orientation(Math::Quaternion::Identity()),
          angularVelocity(Math::Vector3::Zero()), torqueAccum(Math::Vector3::Zero()),
          inverseInertia(Math::Matrix3::Identity()), inverseInertiaWorld(Math::Matrix3::Identity())
    {}

    void RigidBody::SetMass(Scalar newMass)
    {
        mass = newMass;
        inverseMass = (mass != 0.0f) ? 1.0f / mass : 0.0f;
    }

    void RigidBody::ApplyForce(const Math::Vector3& force)
    {
        forceAccum += force;
    }

    void RigidBody::ApplyForceAtPoint(const Math::Vector3& force, const Math::Vector3& point)
    {
        // 力作用点相对于质心的向量
        Math::Vector3 r = point - position;
        ApplyForce(force);
        ApplyTorque(Math::Vector3::Cross(r, force));
    }

    void RigidBody::ApplyTorque(const Math::Vector3& torque)
    {
        torqueAccum += torque;
    }

    void RigidBody::Update(Scalar deltaTime)
    {
        // 更新线速度（半隐式欧拉法）
        Math::Vector3 acceleration = forceAccum * inverseMass;
        velocity += acceleration * deltaTime;

        // 更新位置
        position += velocity * deltaTime;

        // 更新角速度
        UpdateInertiaTensor(); // 更新世界坐标系下的惯性张量
        Math::Vector3 angularAcceleration = inverseInertiaWorld * torqueAccum;
        angularVelocity += angularAcceleration * deltaTime;

        // 更新旋转（四元数积分）
        Math::Quaternion deltaRotation(0.0f, angularVelocity.x * deltaTime, angularVelocity.y * deltaTime, angularVelocity.z * deltaTime);
        orientation = orientation + deltaRotation * orientation * 0.5f;
        orientation.Normalize();

        ClearAccumulators();
    }

    void RigidBody::ClearAccumulators()
    {
        forceAccum = Math::Vector3::Zero();
        torqueAccum = Math::Vector3::Zero();
    }

    void RigidBody::UpdateInertiaTensor()
    {
        // 将局部惯性张量转换到世界坐标系
        Math::Matrix3 rotationMatrix = orientation.ToRotationMatrix();
        inverseInertiaWorld = rotationMatrix * inverseInertia * rotationMatrix.Transpose();
    }
}