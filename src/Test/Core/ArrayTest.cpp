#include "Test/Core/ArrayTest.h"
#include <cassert>

using Polaris::Array;

void Polaris::TestArray()
{
    PLS_INFO("==== CPU Array Test ====");

    // ����һ����СΪ 10 �� Array������� 5
    Array<int, 10> cpuArray;
    cpuArray.fill(5);

    // ��֤����Ԫ�ؾ�Ϊ 5��ʹ�� operator[]��
    for (size_t i = 0; i < cpuArray.size(); ++i) {
        assert(cpuArray[i] == 5);
    }

    // ���Ե���������
    for (auto it = cpuArray.begin(); it != cpuArray.end(); ++it) {
        assert(*it == 5);
    }

    // �޸ĵ�һ��Ԫ�ز���֤
    cpuArray[0] = 100;
    assert(cpuArray[0] == 100);

    PLS_INFO("CPU Array test passed");
}