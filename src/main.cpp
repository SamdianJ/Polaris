#include "Test/Core/MathTest.h"
#include "Test/Core/AllocatorTest.h"
#include "Test/Core/StorageTest.h"
#include "Test/Core/ArrayTest.h"
#include "Test/Core/MatrixTest.h"
#include "Polaris.h"

int main() {
    Polaris::MathTest();
    Polaris::Device::Test_Allocator();
    Polaris::TestCPUStorage();
    Polaris::Device::TestGPUStorage();
    Polaris::TestArray();
    Polaris::Device::TestArray();
    Polaris::MatrixTest();
    Polaris::Device::MatrixTest();
    PLS_DEBUG("Pi: {}", Polaris::PI);
}
