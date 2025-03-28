#include "Test/Core/ArrayTest.h"
#include <cassert>

using Polaris::Array;

void Polaris::TestArray()
{
    PLS_INFO("==== CPU Array Test ====");

    // 创建一个大小为 10 的 Array，并填充 5
    Array<int, 10> cpuArray;
    cpuArray.fill(5);

    // 验证所有元素均为 5（使用 operator[]）
    for (size_t i = 0; i < cpuArray.size(); ++i) {
        assert(cpuArray[i] == 5);
    }

    // 测试迭代器遍历
    for (auto it = cpuArray.begin(); it != cpuArray.end(); ++it) {
        assert(*it == 5);
    }

    // 修改第一个元素并验证
    cpuArray[0] = 100;
    assert(cpuArray[0] == 100);

    PLS_INFO("CPU Array test passed");
}