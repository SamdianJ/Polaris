#include "Test/Core/MathTest.h"
#include "Test/Core/AllocatorTest.h"
#include "Test/Core/StorageTest.h"
#include "Polaris.h"

int main() {
    Polaris::MathTest();
    Polaris::Device::Test_Allocator();
    Polaris::TestCPUStorage();
    Polaris::Device::TestGPUStorage();
    PLS_DEBUG("Pi: {}", Polaris::PI);
}
