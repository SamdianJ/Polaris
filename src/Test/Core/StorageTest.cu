
#include "Test/Core/StorageTest.h"

using namespace Polaris;

#ifdef __CUDACC__
void Polaris::Device::TestGPUStorage() {
    std::cout << "==== GPU Storage ���� ====" << std::endl;
    Storage<int, Platform::CUDA> gpuStorage;

    // ���� CPU �ϴ�������
    Storage<int, Platform::CPU> cpuStorage;
    cpuStorage.PushBack(1);
    cpuStorage.PushBack(2);
    cpuStorage.PushBack(3);
    cpuStorage.PushBack(4);

    // ���䵽 GPU
    gpuStorage.Transfer(cpuStorage);
    PLS_INFO("Load data from host to deviece");

    PLS_INFO("device copy");
    auto gpuStorageCopy(gpuStorage);

    // ����� CPU ���м��
    Storage<int, Platform::CPU> checkStorage = gpuStorage.LoadToHost(gpuStorageCopy);
    PLS_INFO("Load data from device to host");
    checkStorage.Print(true);
    PLS_INFO("Storage device test success");;
}
#endif