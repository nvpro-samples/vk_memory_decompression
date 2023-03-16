cmake_minimum_required(VERSION 3.5)

get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

_add_project_definitions(${PROJNAME})

#####################################################################################
# additions from packages needed for this sample
# add refs  in LIBRARIES_OPTIMIZED
# add refs  in LIBRARIES_DEBUG
# add files in PACKAGE_SOURCE_FILES

set( EXENAME ${PROJNAME} )
_add_package_VulkanSDK()

#####################################################################################
# Optional libdeflate
# TODO: move to third party
macro(_add_package_LIBDEFLATE)
  get_directory_property(hasParent PARENT_DIRECTORY)
  # This uses a slightly different structure than nvpro_core's if(hasParent) / else
  # block, so that USING_LIBDEFLATE gets set in this file if we're running build_all.
  # Otherwise, USING_LIBDEFLATE only gets set in the build_all scope, libdeflate
  # isn't included, and the build fails. If this is moved into nvpro_core, this
  # branch can change back to the standard structure.
  set(USING_LIBDEFLATE ON)
  if(hasParent)
    set(USING_LIBDEFLATE ON PARENT_SCOPE)
  endif()
  LIST(APPEND LIBRARIES_OPTIMIZED libdeflate_static)
  LIST(APPEND LIBRARIES_DEBUG libdeflate_static)
endmacro()
#####################################################################################

_add_package_LIBDEFLATE()

#libdeflate
if (USING_LIBDEFLATE)
  if (NOT TARGET libdeflate_static AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libdeflate)
    include(libdeflatestatic.cmake)
  endif()
endif()

#####################################################################################
# process the rest of some cmake code that needs to be done *after* the packages add
_add_nvpro_core_lib()

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
file(GLOB GLSL_FILES *.glsl)

#####################################################################################
# Executable
#
if(WIN32 AND NOT GLUT_FOUND)
  add_definitions(/wd4996) #remove printf warning
  add_definitions(/wd4244) #remove double to float conversion warning
  add_definitions(/wd4305) #remove double to float truncation warning
else()
  add_definitions(-fpermissive)
endif()
add_executable(${EXENAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
_set_subsystem_console(${EXENAME})

#####################################################################################
# common source code needed for this sample
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES}
)
source_group("Source Files" FILES ${SOURCE_FILES})

if(UNIX)
  set(UNIXLINKLIBS dl pthread)
else()
  set(UNIXLINKLIBS)
endif()

#####################################################################################
# Linkage
#
target_link_libraries(${EXENAME} ${PLATFORM_LIBRARIES} nvpro_core)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${EXENAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${EXENAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#

_finalize_target( ${EXENAME} )


