# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "CCF.h.in"
  ${PROJECT_BINARY_DIR}/CCF.h
  )
 
# add the binary tree to the search path for include files
# so that we will find CCF.h
include_directories("${PROJECT_BINARY_DIR}")

FILE(GLOB folder_source *.c*)
  FILE(GLOB folder_header *.h)
  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header} CCF.h)

add_library(ccf ${LIB_TYPE} ${folder_source} ${folder_header})

if(USE_TBB)
SET_TARGET_PROPERTIES(ccf PROPERTIES LINK_FLAGS -Wl,${CCF_LINK_FLAGS})
endif()

target_link_libraries (ccf ${LIBS})

install (TARGETS ccf DESTINATION lib)
install (FILES ${folder_header} ${PROJECT_BINARY_DIR}/CCF.h DESTINATION "include/ccf")

set(LIBS ${LIBS} ccf)