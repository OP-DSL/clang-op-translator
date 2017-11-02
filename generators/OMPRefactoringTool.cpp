#include "OMPRefactoringTool.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace OP2 {

OMPRefactoringTool::OMPRefactoringTool(
    const clang::tooling::CompilationDatabase &Compilations,
    const ParLoop &loop,
    std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps)
    : OP2KernelGeneratorBase(
          Compilations,
          {std::string(SKELETONS_DIR) + skeletons[loop.getKernelType()]}, loop,
          OMPRefactoringTool::_postfix, PCHContainerOps),
      ompKernelHandler(&getReplacements(), loop),
      seqKernelHandler(&getReplacements(), loop) {}

void OMPRefactoringTool::addGeneratorSpecificMatchers(
    clang::ast_matchers::MatchFinder &Finder) {
  Finder.addMatcher(OMPKernelHandler::locRedVarMatcher, &ompKernelHandler);
  Finder.addMatcher(OMPKernelHandler::locRedToArgMatcher, &ompKernelHandler);
  Finder.addMatcher(OMPKernelHandler::ompParForMatcher, &ompKernelHandler);
  Finder.addMatcher(SeqKernelHandler::userFuncMatcher, &seqKernelHandler);
  Finder.addMatcher(SeqKernelHandler::funcCallMatcher, &seqKernelHandler);
  Finder.addMatcher(SeqKernelHandler::opMPIReduceMatcher, &seqKernelHandler);
  Finder.addMatcher(SeqKernelHandler::mapIdxDeclMatcher, &seqKernelHandler);
}

} // namespace OP2
