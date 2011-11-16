# add our sample(s)

FILE(GLOB folder_source *.c*)
  FILE(GLOB folder_header *.h*)
  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})

include_directories("${PROJECT_BINARY_DIR}")

#set(folder_header  ${folder_header})

# ccfCam
add_executable(ccfCam ccfCam.cpp )
if(USE_TBB)
SET_TARGET_PROPERTIES(ccfCam PROPERTIES LINK_FLAGS -Wl,${CCF_LINK_FLAGS})
endif()
target_link_libraries (ccfCam ${LIBS})

# ccfSequence
add_executable(ccfSequence ccfSequence.cpp ${folder_header})
if(USE_TBB)
SET_TARGET_PROPERTIES(ccfSequence PROPERTIES LINK_FLAGS -Wl,${CCF_LINK_FLAGS})
endif()
target_link_libraries (ccfSequence ${LIBS})

# freshTrainer
add_executable(freshTrainer freshTrainer.cpp ${folder_header})
if(USE_TBB)
SET_TARGET_PROPERTIES(freshTrainer PROPERTIES LINK_FLAGS -Wl,${CCF_LINK_FLAGS})
endif()
target_link_libraries (freshTrainer ${LIBS})

# add the install targets
install (TARGETS ccfCam DESTINATION bin)
install (TARGETS ccfSequence DESTINATION bin)
install (TARGETS freshTrainer DESTINATION bin)