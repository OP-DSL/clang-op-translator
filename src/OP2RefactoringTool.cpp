#include "OP2RefactoringTool.hpp"
#include "OPDataRegister.hpp"
#include "ParLoopHandler.hpp"
#include "generators/MasterkernelGenerator.hpp"

namespace OP2 {
OP2RefactoringTool::OP2RefactoringTool(
    std::vector<std::string> &commandLineArgs,
    clang::tooling::FixedCompilationDatabase &compilations,
    clang::tooling::CommonOptionsParser &optionsParser, OP2Targets opTarget,
    OP2Optimizations optimizationFlags)
    : OP2WriteableRefactoringTool(compilations,
                                  optionsParser.getSourcePathList()),
      opTarget(opTarget), optimizationFlags(optimizationFlags),
      commandLineArgs(commandLineArgs), Compilations(compilations),
      application() {
  std::string applicationName = optionsParser.getSourcePathList()[0];
  size_t basename_start = applicationName.rfind("/"),
         basename_end = applicationName.rfind(".");
  if (basename_start == std::string::npos) {
    basename_start = 0;
  } else {
    basename_start++;
  }
  if (basename_end == std::string::npos || basename_end < basename_start)
    llvm::errs() << "Invalid applicationName: " << applicationName << "\n";
  applicationName =
      applicationName.substr(basename_start, basename_end - basename_start);
  application.setName(applicationName);
}
void OP2RefactoringTool::generateKernelFiles() {
  if (opTarget == none)
    return;
  if (opTarget == seq || opTarget == all) {
    SeqGenerator generator(application, commandLineArgs, Compilations,
                           optimizationFlags);
    generator.generateKernelFiles();
  }
  if (opTarget == openmp || opTarget == all) {
    OpenMPGenerator generator(application, commandLineArgs, Compilations,
                              optimizationFlags);
    generator.generateKernelFiles();
  }
  if (opTarget == openmp4 || opTarget == all) {
    OpenMP4Generator generator(application, commandLineArgs, Compilations,
                              optimizationFlags);
    generator.generateKernelFiles();
  }
  if (opTarget == vec || opTarget == all) {
    VectorizedGenerator generator(application, commandLineArgs, Compilations,
                                  optimizationFlags, "skeleton_veckernels.cpp");
    generator.generateKernelFiles();
  }
  if (opTarget == cuda || opTarget == all) {
    CUDAGenerator generator(application, commandLineArgs, Compilations,
                            optimizationFlags, "skeleton_kernels.cu");
    generator.generateKernelFiles();
  }
}

int OP2RefactoringTool::generateOPFiles() {
  clang::ast_matchers::MatchFinder Finder;
  OPParLoopDeclarator callback(*this);
  // creating and setting callback to handle op_par_loops
  ParLoopHandler parLoopHandlerCallback(*this, application, callback);
  Finder.addMatcher(
      callExpr(callee(functionDecl(hasName("op_par_loop")))).bind("par_loop"),
      &parLoopHandlerCallback);

  DataRegister drCallback(application, &getReplacements(), optimizationFlags);
  drCallback.addToFinder(Finder);

  return run(
      clang::tooling::newFrontendActionFactory(&Finder, &callback).get());
}

std::string
OP2RefactoringTool::getOutputFileName(const clang::FileEntry *Entry) const {
  llvm::outs() << "Rewrite buffer for file: " << Entry->getName() << "\n";

  std::string filename = Entry->getName().str();
  size_t basename_start = filename.rfind("/") + 1,
         basename_end = filename.rfind(".");
  if (basename_start == std::string::npos)
    basename_start = 0;
  if (basename_end == std::string::npos || basename_end < basename_start)
    llvm::errs() << "Invalid filename: " << Entry->getName() << "\n";
  filename = filename.substr(basename_start, basename_end - basename_start) +
             "_op.cpp";
  return filename;
}

} // namespace OP2
