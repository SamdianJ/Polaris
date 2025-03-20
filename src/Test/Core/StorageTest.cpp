#include <iostream>
#include "Test/Core/StorageTest.h"

using namespace Polaris;

// CPU 端测试
void Polaris::TestCPUStorage()
{
    std::cout << "==== CPU Storage 测试 ====" << std::endl;
    Storage<int, Platform::CPU> cpuStorage;

    // 插入数据
    cpuStorage.PushBack(10);
    cpuStorage.PushBack(20);
    cpuStorage.PushBack(30);
    cpuStorage.Print();

    // 扩容 & 重新调整大小
    cpuStorage.Reserve(10);
    cpuStorage.Resize(5);
    cpuStorage.Print(true);

    // 迭代器测试
    std::cout << "遍历 CPU Storage: ";
    for (auto it = cpuStorage.Begin(); it != cpuStorage.End(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "copy storage " << std::endl;
    auto cpuStorageCopy(cpuStorage);
    cpuStorageCopy.Print();
}




