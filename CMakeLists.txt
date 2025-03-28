cmake_minimum_required(VERSION 3.18)
project(PhysicsEngine VERSION 1.0 LANGUAGES CXX CUDA)

include(cmake/config.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)

# GetFiles
GetIncludes(PolarisInc)
GetSources(PolarisSrc)
GetCudaIncludes(PolarisCudaInc)
GetCudaSources(PolarisCudaSrc)
list(APPEND PolarisInc ${PolarisCudaInc})
list(APPEND PolarisSrc ${PolarisCudaSrc})
GetIncludeDirectories(IncludeDirectories)

# main
file(GLOB MainSrc ${CMAKE_SOURCE_DIR}/src/main.cpp)

# lib
message(STATUS "Build For PolarisLib ...")
SourceGroupByDir(PolarisInc "Header Files/")
SourceGroupByDir(PolarisSrc "Source Files/")
add_library(PolarisLib STATIC ${PolarisInc} ${PolarisSrc})
# target properties
set_target_properties(PolarisLib PROPERTIES OUTPUT_NAME "Polaris")
# includes
target_include_directories(PolarisLib PUBLIC ${IncludeDirectories})

message(STATUS "Build For PolarisLib Finish")

message(STATUS "# Build For PolarisExe ...")
message(STATUS "MainCpp: ${MainSrc}")
add_executable(PolarisExe ${MainSrc})
message(STATUS "Build For PolarisExe Finish")
set_target_properties(PolarisExe PROPERTIES OUTPUT_NAME "PolarisExe")

# dependencies
add_dependencies(PolarisExe PolarisLib)
# link libs

# thirdparty
add_subdirectory(thirdparty/fmt)
target_include_directories(PolarisExe PRIVATE thirdparty/fmt/include)

# link
target_link_libraries(PolarisExe PolarisLib fmt::fmt CUDA::cudart)
target_include_directories(PolarisExe PRIVATE ${IncludeDirectories})

