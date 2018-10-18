############################################################
#base
############################################################

cmake_minimum_required (VERSION 3.4)

############################################################
#setting Project informations
############################################################

set(PROJECT_NAME op2-clang)

set(LIBRARY_LIST clangFrontend  clangSerialization clangDriver clangParse clangSema)
set(LIBRARY_LIST ${LIBRARY_LIST} clangAnalysis clangEdit clangAST clangLex clangBasic clangTooling clangToolingCore clangFormat clangRewrite clangASTMatchers clangToolingInclusions)

############################################################
#generate makefiles
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

#ugly way to find clang system headers..it's not hardcoded and works if clang is installed.. but only on linux..
execute_process(COMMAND bash -c "clang++ '-###' -c ${CMAKE_CURRENT_SOURCE_DIR}/src/op2.cpp 2>&1 | tr ' ' '\n'" 
                COMMAND grep -A1 resource
                COMMAND grep /  
                OUTPUT_VARIABLE CLANG_SYSTEM_HEADERS)
message(STATUS "clang system headers found in: ${CLANG_SYSTEM_HEADERS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OP2_INSTALL_PATH "$ENV{OP2_INSTALL_PATH}")

set(SKELETONS_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/skeletons/\")

if(OP2_INSTALL_PATH)
  set(OP2_INC \"${OP2_INSTALL_PATH}/c/include/\")
else()
  message(STATUS "OP2_INSTALL_PATH not found. Use default: \"OP2-Clang/../OP2-Common/op2/c/include/\"")
  set(OP2_INC \"${CMAKE_CURRENT_SOURCE_DIR}/../OP2-Common/op2/c/include/\")
endif()

link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(
  -DSKELETONS_DIR=${SKELETONS_DIR}
  -DOP2_INC=${OP2_INC}
  -DCLANG_SYSTEM_HEADERS=${CLANG_SYSTEM_HEADERS}
  )

add_executable(${PROJECT_NAME}
  src/op2.cpp
  src/core/OPParLoopData.cpp
  src/AppFileTransformations.cpp
  src/AppFileRefactoringTool.cpp
  )

#src / OPAppFileRefactoringTool.cpp
add_executable(op-check
  src/op-check.cpp
  src/core/OPParLoopData.cpp
  )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

target_link_libraries(${PROJECT_NAME} ${LIBRARY_LIST})

target_link_libraries(${PROJECT_NAME}
  LLVMTarget
  LLVMOption # Support
  )

target_link_libraries(op-check ${LIBRARY_LIST})

target_link_libraries(op-check
  LLVMTarget
  LLVMOption # Support
  )
