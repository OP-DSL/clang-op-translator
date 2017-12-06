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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${LLVM_INCLUDE_DIRS})
if(LLVM_BUILD_MAIN_SRC_DIR)
  include_directories(${LLVM_BUILD_MAIN_SRC_DIR}/tools/clang/include)
  include_directories(${LLVM_BUILD_BINARY_DIR}/tools/clang/include)
endif()

set(OP2_INSTALL_PATH "$ENV{OP2_INSTALL_PATH}")

set(SKELETONS_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/skeletons/\")

if(OP2_INSTALL_PATH)
  set(OP2_INC \"${OP2_INSTALL_PATH}/c/include/\")
else()
  message(STATUS "OP2_INSTALL_PATH not found. Use default: \"~/OP2-Common/op2/c/include/\"")
  set(OP2_INC \"~/OP2-Common/op2/c/include/\")
endif()

link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(
  -DSKELETONS_DIR=${SKELETONS_DIR}
  -DOP2_INC=${OP2_INC}
  )

add_executable(${PROJECT_NAME}
  src/op2.cpp
  src/ParLoopHandler.cpp
  src/core/utils.cpp
  src/core/OPParLoopData.cpp
  src/generators/common/BaseKernelHandler.cpp
  src/generators/sequential/SeqKernelHandler.cpp
  src/generators/openmp/OMPKernelHandler.cpp
  src/generators/vectorization/VecKernelHandler.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-exceptions -fno-rtti")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

target_link_libraries(${PROJECT_NAME} ${LIBRARY_LIST})

target_link_libraries(${PROJECT_NAME}
  LLVMTarget
  LLVMOption # Support
  )

