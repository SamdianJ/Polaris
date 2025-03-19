#include "Test/Core/MathTest.h"
#include "Test/Core/AllocatorTest.h"
#include "Polaris.h"

int main() {
    Polaris::MathTest();
    Polaris::Device::Test_Allocator();
    PLS_DEBUG("Pi: {}", Polaris::PI);
}
