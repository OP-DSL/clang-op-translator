set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(op2
  op2.cpp
  ParLoopHandler.cpp
  ParLoopHandler.h
  OP2RefactoringTool.hpp
  utils.cpp
  utils.h
  OPParLoopData.h
  OPParLoopData.cpp
  generators/SeqRefactoringTool.h
  generators/SeqRefactoringTool.cpp
  generators/BaseKernelHandler.h
  generators/BaseKernelHandler.cpp
  generators/SeqKernelHandler.h
  generators/SeqKernelHandler.cpp
  )

target_link_libraries(op2
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  )
