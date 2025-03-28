cmake_minimum_required(VERSION 3.18)
project(PhysicsEngine VERSION 1.0 LANGUAGES CXX CUDA)

option(ENABLE_DEBUG "Enable debugging" OFF)
option(ENABLE_MULTITHREAD "Enable multithreading support" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(include)
