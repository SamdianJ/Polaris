#include <iostream>
#include "Test/Core/StorageTest.h"

using namespace Polaris;

// CPU �˲���
void Polaris::TestCPUStorage()
{
    std::cout << "==== CPU Storage ���� ====" << std::endl;
    Storage<int, Platform::CPU> cpuStorage;

    // ��������
    cpuStorage.PushBack(10);
    cpuStorage.PushBack(20);
    cpuStorage.PushBack(30);
    cpuStorage.Print();

    // ���� & ���µ�����С
    cpuStorage.Reserve(10);
    cpuStorage.Resize(5);
    cpuStorage.Print(true);

    // ����������
    std::cout << "���� CPU Storage: ";
    for (auto it = cpuStorage.Begin(); it != cpuStorage.End(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "copy storage " << std::endl;
    auto cpuStorageCopy(cpuStorage);
    cpuStorageCopy.Print();
}




