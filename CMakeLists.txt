cmake_minimum_required (VERSION 2.6)

project (CCF_Library)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${OpenCV_LIBS})

#SET( CMAKE_INSTALL_PREFIX . CACHE TYPE “description” )

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Allow the developer to select if Dynamic or Static libraries are built
OPTION (BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
# Set the LIB_TYPE variable to STATIC
SET (LIB_TYPE STATIC)
IF (BUILD_SHARED_LIBS)
  # User wants to build Dynamic Libraries, so change the LIB_TYPE variable to CMake keyword 'SHARED'
  SET (LIB_TYPE SHARED)
ENDIF (BUILD_SHARED_LIBS)

# use Intel Threading Building Blocks?
option (USE_TBB "Use Intel TBB" OFF)

# use alternative filterchain ?
option (USE_RT "Use alternative trajectory filter" OFF)

# build evaluation program
OPTION (BUILD_EVALUATION "Build Evaluation Program" ON)

if (USE_TBB)
add_definitions(-DUSE_TBB)

SET( TBB_INCLUDE_DIR /opt/local/include CACHE TYPE “description” )
SET( TBB_LIB_DIR /opt/local/lib CACHE TYPE “description” )

include_directories(${TBB_INCLUDE_DIR})
include_directories(${TBB_INCLUDE_DIR} "/..")

set(LIBS ${LIBS} tbb tbbmalloc)

if(NOT APPLE)
set(LIBS ${LIBS} tbbmalloc_proxy)
endif()

set(CCF_LINK_FLAGS ${CCF_LINK_FLAGS} -L${TBB_LIB_DIR})

else()
remove_definitions(-DUSE_TBB)
set(LIBS ${OpenCV_LIBS})

endif(USE_TBB)

if(USE_RT)
add_definitions(-DUSE_RT)
include_directories ("${PROJECT_SOURCE_DIR}/ccf_rt")
add_subdirectory (ccf_rt)
set(LIBS ${LIBS} ccf_rt)

else()
remove_definitions(-DUSE_RT)
include_directories ("${PROJECT_SOURCE_DIR}/ccf")
add_subdirectory (ccf)
set(LIBS ${LIBS} ccf)
endif()

# sample directory
add_subdirectory (sample)

#evaluation code
if(BUILD_EVALUATION)
add_subdirectory (evaluation)
endif()


