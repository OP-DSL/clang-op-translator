#include "OMPRefactoringTool.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace OP2 {

OMPRefactoringTool::OMPRefactoringTool(
    const clang::tooling::CompilationDatabase &Compilations,
    const ParLoop &loop,
    std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps)
    : OP2KernelGeneratorBase(Compilations, {skeletons[loop.getKernelType()]},
                             loop, "kernel", PCHContainerOps),
      ompKernelHandler(&getReplacements(), loop) {}

void OMPRefactoringTool::addGeneratorSpecificMatchers(
    clang::ast_matchers::MatchFinder &Finder) {
  Finder.addMatcher(OMPKernelHandler::locRedVarMatcher, &ompKernelHandler);
}

} // namespace OP2
