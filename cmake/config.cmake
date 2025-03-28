# Group
macro(SourceGroupByDir TargetFiles TargetSpace)
    if(MSVC AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(GitDir ${CMAKE_SOURCE_DIR})
        foreach(TargetFile ${${TargetFiles}})
            # message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
            string(REGEX REPLACE ${GitDir}/\(.*\) \\1 RelativePath ${TargetFile})
            string(REGEX REPLACE "\(.*\)/.*" \\1 TargetGroupName ${RelativePath})
            string(COMPARE EQUAL ${RelativePath} ${TargetGroupName} HaveNoGroupPath)
            string(REPLACE "/" "\\" TargetGroupName ${TargetGroupName})
            if(HaveNoGroupPath)
                set(TargetGroupName "NoGroup")
            endif(HaveNoGroupPath)
            
            string(APPEND ${TargetSpace} ${TargetGroupName})
            # message("TargetFile: ${TargetFile}\n TargetSpace: ${TargetSpace} TargetGroupName:${TargetGroupName}")
            source_group(${TargetSpace}${TargetGroupName} FILES ${TargetFile})
        endforeach(TargetFile)
    endif()
endmacro(SourceGroupByDir)

# ####################################################
# Include and Source
macro(GetIncludes PolarisInc)
    file(GLOB_RECURSE ${PolarisInc}
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/src/*.h
        ${CMAKE_SOURCE_DIR}/thirdparty/*.h)
endmacro(GetIncludes)

macro(GetCudaIncludes PolarisCudaInc)
    file(GLOB_RECURSE ${PolarisCudaInc}
        ${CMAKE_SOURCE_DIR}/include/*.cuh
        ${CMAKE_SOURCE_DIR}/src/*.cuh
        ${CMAKE_SOURCE_DIR}/thirdparty/*.cuh)
endmacro(GetCudaIncludes)

macro(GetSources PolarisSrc)
    file(GLOB_RECURSE ${PolarisSrc}
        include/*.cpp
        src/*.cpp)
endmacro(GetSources)

macro(GetCudaSources PolarisCudaSrc)
    file(GLOB_RECURSE ${PolarisCudaSrc}
        include/*.cu
        src/*.cu)
endmacro(GetCudaSources)

macro(GetIncludeDirectories InDirs)
    set(${InDirs})
    list(APPEND ${InDirs} ${CMAKE_SOURCE_DIR})
    list(APPEND ${InDirs} ${CMAKE_SOURCE_DIR}/include)
    list(APPEND ${InDirs} ${CMAKE_SOURCE_DIR}/include/Core)
    list(APPEND ${InDirs} ${CMAKE_SOURCE_DIR}/include/Platform)
    list(APPEND ${InDirs} ${CMAKE_SOURCE_DIR}/include/Rigid)
    list(APPEND ${InDirs} ${CMAKE_SOURCE_DIR}/include/Test)
    list(APPEND ${InDirs} ${CMAKE_SOURCE_DIR}/thirdparty)
endmacro(GetIncludeDirectories)

# CompileOptions
function(SetTargetWarningFlags TargetProject)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
            ${TargetProject} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Werror
            -Wno-deprecated
            -Wno-misleading-indentation
            -Wno-unused
            -Wno-unknown-pragmas
            -Wno-unknown-warning-option
            -Wno-unused-command-line-argument>
        )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            ${TargetProject} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Werror
            # -Wsuggest-override
            -Wreturn-type
            -Wno-deprecated
            -Wno-misleading-indentation
            -Wno-unused
            -Wno-unused-result
            -Wno-sign-compare
            -Wno-class-memaccess
            -Wno-unknown-pragmas
            -Wno-comment
            >
        )
        target_compile_options(
            ${TargetProject} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=-Wall
            -Xcompiler=-Werror
            -Xcompiler=-Wno-deprecated
            -Xcompiler=-Wno-misleading-indentation
            -Xcompiler=-Wno-unused
            -Xcompiler=-Wno-sign-compare
            -Xcompiler=-Wno-class-memaccess
            -Xcompiler=-Wno-unknown-pragmas
            -Xcompiler=-Wno-comment
            >
        )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(
            ${TargetProject} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            /WX /W2
            # not ANSI encode
            /wd4819 #1
            # dllexport
            /wd4251 #2
            /wd4275 #2
            # cast lose precision
            /wd4244 #2
            /wd4305 #1
            # unknown pragma
            /wd4068 #1
            # deprecated
            /wd4996 #3
            >
        )
        target_compile_options(
            ${TargetProject} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/WX
            -Xcompiler=/W2
            -Xcompiler=/wd4819
            -Xcompiler=/wd4251
            -Xcompiler=/wd4275
            -Xcompiler=/wd4244
            -Xcompiler=/wd4305
            -Xcompiler=/wd4068
            -Xcompiler=/wd4996
            -Xcompiler=/wd4005 #macro defined twice
            >
        )
    endif()
endfunction(SetTargetWarningFlags)