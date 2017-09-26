set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(op2
  op2.cpp
  ParLoopHandler.cpp
  ParLoopHandler.h
  utils.cpp
  utils.h
  op_par_loop.cpp
  op_par_loop.h
  )

target_link_libraries(op2
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  )
