# add our sample(s)

FILE(GLOB folder_source *.c*)
  FILE(GLOB folder_header *.h*)
  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})

include_directories("${PROJECT_BINARY_DIR}")

#set(folder_header  ${folder_header})

# ccfEvaluation
add_executable(ccfEvaluation ccfEvaluation.cpp ${folder_header})
if(USE_TBB)
SET_TARGET_PROPERTIES(ccfEvaluation PROPERTIES LINK_FLAGS -Wl,${CCF_LINK_FLAGS})
endif()
target_link_libraries (ccfEvaluation ${LIBS})

# add the install targets
install (TARGETS ccfEvaluation DESTINATION bin)