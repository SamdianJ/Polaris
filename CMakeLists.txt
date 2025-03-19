cmake_minimum_required(VERSION 3.10)

if(MSVC)
    set(CMAKE_GENERATOR_TOOLSET "v142" CACHE STRING "Platform Toolset" FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(PhysicsEngine VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "50")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CORE_SOURCES "src/Core/*.cpp")
file(GLOB_RECURSE RIGID_SOURCES "src/Rigid/*.cpp")
file(GLOB_RECURSE PLATFORM_SOURCES "src/Platform/*.cpp")
file(GLOB_RECURSE TEST_SOURCES "src/Test/*.cpp")
file(GLOB_RECURSE MAIN_SOURCES "src/main.cpp")
file(GLOB_RECURSE CORE_CUDA_SOURCES "src/Core/*.cu")
file(GLOB_RECURSE RIGID_CUDA_SOURCES "src/Rigid/*.cu")
file(GLOB_RECURSE PLATFORM_CUDA_SOURCES "src/Platform/*.cu")
file(GLOB_RECURSE TEST_CUDA_SOURCES "src/Test/*.cu")
file(GLOB_RECURSE CORE_HEADS "include/Core/*.h")
file(GLOB_RECURSE RIGID_HEADS "include/Rigid/*.h")
file(GLOB_RECURSE PLATFORM_HEADS "include/Platform/*.h")
file(GLOB_RECURSE TEST_HEADS "include/Test/*.h")

set(CUDA_SOURCES 
    ${CORE_CUDA_SOURCES}
    ${RIGID_CUDA_SOURCES}
    ${PLATFORM_CUDA_SOURCES}
    ${TEST_CUDA_SOURCES}
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(Core STATIC ${CORE_SOURCES} ${CORE_CUDA_SOURCES} ${CORE_HEADS})
add_library(Rigid STATIC ${RIGID_SOURCES} ${RIGID_CUDA_SOURCES} ${RIGID_HEADS})
add_library(Platform STATIC ${PLATFORM_SOURCES} ${PLATFORM_CUDA_SOURCES} ${PLATFORM_HEADS})
add_library(Test STATIC ${TEST_SOURCES} ${TEST_CUDA_SOURCES} ${TEST_HEADS})
add_executable(PhysicsEngine ${MAIN_SOURCES})

add_subdirectory(thirdparty/fmt)

target_include_directories(Core PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_SOURCE_DIR}/include/Core>
)
target_include_directories(Rigid PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_SOURCE_DIR}/include/Rigid>
)
target_include_directories(Platform PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_SOURCE_DIR}/include/Platform>
)
target_include_directories(Test PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_SOURCE_DIR}/include/Test>
)
target_include_directories(PhysicsEngine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(PhysicsEngine PRIVATE thirdparty/fmt/include)

target_link_libraries(PhysicsEngine 
    PRIVATE 
    Core 
    Rigid 
    Platform 
    Test
    CUDA::cudart
    fmt::fmt
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(PhysicsEngine PRIVATE -Wall -Wextra -O2)
    target_compile_options(Core PRIVATE -Wall -Wextra -O2)
    target_compile_options(Rigid PRIVATE -Wall -Wextra -O2)
    target_compile_options(Platform PRIVATE -Wall -Wextra -O2)
    
    target_compile_options(PhysicsEngine PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O2>
    )
endif()

if (MSVC)
    target_compile_options(PhysicsEngine PRIVATE 
        $<$<CONFIG:Debug>:/W4;/RTC1>
    )
    target_compile_options(Core PRIVATE 
        $<$<CONFIG:Debug>:/W4;/RTC1>
    )
    target_compile_options(Rigid PRIVATE 
        $<$<CONFIG:Debug>:/W4;/RTC1>
    )
    target_compile_options(Platform PRIVATE 
        $<$<CONFIG:Debug>:/W4;/RTC1>
    )
    
    target_compile_options(PhysicsEngine PRIVATE 
        $<$<CONFIG:Release>:/W4;/O2>
    )
    target_compile_options(Core PRIVATE 
        $<$<CONFIG:Release>:/W4;/O2>
    )
    target_compile_options(Rigid PRIVATE 
        $<$<CONFIG:Release>:/W4;/O2>
    )
    target_compile_options(Platform PRIVATE 
        $<$<CONFIG:Release>:/W4;/O2>
    )
    
    # 对于 CUDA 编译选项，在 MSVC 下也做相应处理
    target_compile_options(PhysicsEngine PRIVATE 
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/W4 -Xcompiler=/RTC1>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/W4 -Xcompiler=/O2>
    )
endif()

function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        file(RELATIVE_PATH _relative_path ${PROJECT_SOURCE_DIR} ${_source})
        get_filename_component(_dir_path ${_relative_path} DIRECTORY)
        if (_dir_path)
            string(REPLACE "/" "\\" _group_path ${_dir_path})
        else()
            set(_group_path "\\")
        endif()
        source_group(${_group_path} FILES ${_source})
    endforeach()
endfunction()

assign_source_group(${CORE_SOURCES} ${CORE_CUDA_SOURCES} ${CORE_HEADS})
assign_source_group(${RIGID_SOURCES} ${RIGID_CUDA_SOURCES} ${RIGID_HEADS})
assign_source_group(${PLATFORM_SOURCES} ${PLATFORM_CUDA_SOURCES}  ${PLATFORM_HEADS}) 
assign_source_group(${TEST_SOURCES} ${TEST_CUDA_SOURCES}  ${TEST_HEADS}) 
assign_source_group(${MAIN_SOURCES})
assign_source_group(${CUDA_SOURCES})

set_target_properties(Core Rigid Platform Test PhysicsEngine PROPERTIES FOLDER "Libraries")
