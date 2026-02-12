# Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.

set(LIBDEF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libdeflate" )

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_PROC)
if(ARCH_PROC MATCHES "^(arm|aarch64)")
    set(ARCH_DIR "arm")
elseif(ARCH_PROC MATCHES "^(x86_64|amd64|i[3-6]86)")
    set(ARCH_DIR "x86")
endif()

set(SOURCES
  ${LIBDEF_DIR}/lib/adler32.c
  ${LIBDEF_DIR}/lib/crc32.c
  ${LIBDEF_DIR}/lib/deflate_compress.c
  ${LIBDEF_DIR}/lib/deflate_decompress.c
  ${LIBDEF_DIR}/lib/gdeflate_compress.c
  ${LIBDEF_DIR}/lib/gdeflate_decompress.c
  ${LIBDEF_DIR}/lib/zlib_compress.c
  ${LIBDEF_DIR}/lib/zlib_decompress.c
  ${LIBDEF_DIR}/lib/utils.c
  ${LIBDEF_DIR}/lib/${ARCH_DIR}/cpu_features.c
)

set(HEADERS
  ${LIBDEF_DIR}/lib/adler32_vec_template.h
  ${LIBDEF_DIR}/lib/bt_matchfinder.h
  ${LIBDEF_DIR}/lib/crc32_table.h
  ${LIBDEF_DIR}/lib/crc32_vec_template.h
  ${LIBDEF_DIR}/lib/decompress_template.h
  ${LIBDEF_DIR}/lib/deflate_compress.h
  ${LIBDEF_DIR}/lib/deflate_constants.h
  ${LIBDEF_DIR}/lib/hc_matchfinder.h
  ${LIBDEF_DIR}/lib/lib_common.h
  ${LIBDEF_DIR}/lib/matchfinder_common.h
  ${LIBDEF_DIR}/lib/unaligned.h
)
file(GLOB ARCH_HEADERS "${LIBDEF_DIR}/lib/${ARCH_DIR}/*.h")
list(APPEND HEADERS ${ARCH_HEADERS})

set(PUBLIC_HEADERS
  ${LIBDEF_DIR}/libdeflate.h
)

include_directories(${LIBDEF_DIR})
add_library(libdeflate_static STATIC ${SOURCES} ${HEADERS} ${PUBLIC_HEADERS})
set_property(TARGET libdeflate_static PROPERTY FOLDER "ThirdParty")
