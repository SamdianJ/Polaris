message(STATUS "Build For PolarisLib ...")

include(config.cmake)

SourceGroupByDir(PolarisInc "Header Files/")
SourceGroupByDir(PolarisSrc "Source Files/")

add_library(PolarisLib STATIC ${PolarisInc} ${PolarisSrc})

# target properties
set_target_properties(PolarisLib PROPERTIES OUTPUT_NAME "Polaris")

# includes
target_include_directories(PolarisLib PUBLIC ${InDirs})
