set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(op2
  op2.cpp
  ParLoopHandler.cpp
  ParLoopHandler.h
  )

target_link_libraries(op2
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  )
