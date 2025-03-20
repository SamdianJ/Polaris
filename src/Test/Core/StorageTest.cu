
#include "Test/Core/StorageTest.h"

using namespace Polaris;

#ifdef __CUDACC__
void Polaris::Device::TestGPUStorage() {
    std::cout << "==== GPU Storage 测试 ====" << std::endl;
    Storage<int, Platform::CUDA> gpuStorage;

    // 先在 CPU 上创建数据
    Storage<int, Platform::CPU> cpuStorage;
    cpuStorage.PushBack(1);
    cpuStorage.PushBack(2);
    cpuStorage.PushBack(3);
    cpuStorage.PushBack(4);

    // 传输到 GPU
    gpuStorage.Transfer(cpuStorage);
    PLS_INFO("Load data from host to deviece");

    PLS_INFO("device copy");
    auto gpuStorageCopy(gpuStorage);

    // 传输回 CPU 进行检查
    Storage<int, Platform::CPU> checkStorage = gpuStorage.LoadToHost(gpuStorageCopy);
    PLS_INFO("Load data from device to host");
    checkStorage.Print(true);
    PLS_INFO("Storage device test success");;
}
#endif