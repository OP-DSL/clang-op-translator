############################################################
#   base
############################################################

cmake_minimum_required (VERSION 3.4)

############################################################
#   setting Project informations
############################################################

set(PROJECT_NAME op2-clang)

set(LIBRARY_LIST clangFrontend  clangSerialization clangDriver clangParse clangSema)
set(LIBRARY_LIST ${LIBRARY_LIST} clangAnalysis clangEdit clangAST clangLex clangBasic clangTooling clangToolingCore clangFormat clangRewrite clangASTMatchers)

############################################################
#   generate makefiles
############################################################

project(${PROJECT_NAME})

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
if(LLVM_BUILD_MAIN_SRC_DIR)
  include_directories(${LLVM_BUILD_MAIN_SRC_DIR}/tools/clang/include)
  include_directories(${LLVM_BUILD_BINARY_DIR}/tools/clang/include)
endif()

set(OP2_INSTALL_PATH "$ENV{OP2_INSTALL_PATH}")

if(OP2_INSTALL_PATH)
  set(SKELETONS_DIR \"${OP2_INSTALL_PATH}/skeletons/\")
else()
  message(STATUS "OP2_INSTALL_PATH not found.")
endif()

link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(
  -DSKELETONS_DIR=${SKELETONS_DIR}
  )

add_executable(${PROJECT_NAME}
  op2.cpp
  ParLoopHandler.cpp
  utils.cpp
  OPParLoopData.cpp
  generators/handler.hpp
  generators/OMPRefactoringTool.cpp
  generators/SeqRefactoringTool.cpp
  generators/BaseKernelHandler.cpp
  generators/SeqKernelHandler.cpp
  generators/OMPKernelHandler.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-exceptions -fno-rtti")


target_link_libraries(${PROJECT_NAME} ${LIBRARY_LIST})

target_link_libraries(${PROJECT_NAME}
  LLVMTarget
  LLVMOption # Support
  )

