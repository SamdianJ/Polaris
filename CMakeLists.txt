# 设置CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

if(MSVC)
  set(CMAKE_GENERATOR_TOOLSET "v142" CACHE STRING "Platform Toolset" FORCE)
endif()

# 项目名称和版本
project(PhysicsEngine VERSION 1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找所有源文件
file(GLOB_RECURSE CORE_SOURCES "src/Core/*.cpp")
file(GLOB_RECURSE RIGID_SOURCES "src/Rigid/*.cpp")
file(GLOB_RECURSE MAIN_SOURCES "src/main.cpp")
file(GLOB_RECURSE CORE_HEADS "include/Core/*.h")
file(GLOB_RECURSE RIGID_HEADS "include/Rigid/*.h")

# 添加Core模块为静态库
add_library(Core STATIC ${CORE_SOURCES} ${CORE_HEADS})

# 添加Rigid模块为静态库
add_library(Rigid STATIC ${RIGID_SOURCES} ${RIGID_HEADS})

# 添加可执行文件
add_executable(PhysicsEngine ${MAIN_SOURCES})

# 包含头文件目录
target_include_directories(Core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(Rigid PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(PhysicsEngine PUBLIC ${PROJECT_SOURCE_DIR}/include)

# 链接Core和Rigid模块
target_link_libraries(PhysicsEngine Core Rigid)

# 设置输出目录（可选）
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 添加编译选项（可选）
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(PhysicsEngine PRIVATE -Wall -Wextra -O2)
    target_compile_options(Core PRIVATE -Wall -Wextra -O2)
    target_compile_options(Rigid PRIVATE -Wall -Wextra -O2)
endif()