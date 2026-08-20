cmake_minimum_required(VERSION 3.22...4.0.0)
project(vk_memory_decompression LANGUAGES C CXX)

# Disable nvpro_core2 sublibraries we don't need
option(NVPRO2_ENABLE_DEFAULT "" OFF)
option(NVPRO2_ENABLE_nvutils "" ON)
option(NVPRO2_ENABLE_nvvk "" ON)

# Call FindNvproCore2 to download nvpro_core2 or find it if it is already
# downloaded, or is located as a subdirectory or up to two levels up.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

###############################################################################
# This sample depends on NVIDIA/libdeflate for CPU compression + decompression.
if(NOT TARGET libdeflate_static)
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libdeflate)
    message(WARNING "Skipping the vk_memory_decompression sample because libdeflate doesn't exist as a submodule at ${CMAKE_CURRENT_SOURCE_DIR}/libdeflate ! You probably need to open a command line in the vk_memory_decompression folder and run `git submodule update --init --recursive`.")
    return()
  endif()
  include(libdeflatestatic.cmake)
endif()

###############################################################################
# Main executable
set(SOURCE_FILES main.cpp)
source_group("Source Files" FILES ${SOURCE_FILES})

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvutils
  nvpro2::nvvk
  libdeflate_static
)

# Make it so this executable is selected by default when opening Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Copy .dll/.so dependencies, if there are any
copy_to_runtime_and_install(${PROJECT_NAME} AUTO)
