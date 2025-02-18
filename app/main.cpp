#include <iostream>

using namespace std;

int main()
{
	std::cout << "Hello, Polaris" << std::endl;
	PhysicsEngine::RigidBody box;
	box.SetMass(2.0f); // 设置质量

	box.ApplyForce(PhysicsEngine::Vector3(0.0f, -9.8f * box.mass, 0.0f));

	// 模拟更新（时间步长 0.016秒 ≈ 1/60帧）
	box.Update(0.016f);

	return 0;
}