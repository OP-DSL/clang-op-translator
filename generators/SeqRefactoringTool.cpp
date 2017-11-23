#include "SeqRefactoringTool.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace OP2 {

SeqRefactoringTool::SeqRefactoringTool(
    const clang::tooling::CompilationDatabase &Compilations,
    const ParLoop &loop,
    std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps)
    : OP2KernelGeneratorBase(Compilations,
                             {std::string(SKELETONS_DIR) + skeletons[0]}, loop,
                             SeqRefactoringTool::_postfix, PCHContainerOps),
      seqKernelHandler(&getReplacements(), loop) {}

void SeqRefactoringTool::addGeneratorSpecificMatchers(
    clang::ast_matchers::MatchFinder &Finder) {
  Finder.addMatcher(SeqKernelHandler::userFuncMatcher, &seqKernelHandler);
  Finder.addMatcher(SeqKernelHandler::funcCallMatcher, &seqKernelHandler);
  Finder.addMatcher(SeqKernelHandler::opMPIReduceMatcher, &seqKernelHandler);
  Finder.addMatcher(SeqKernelHandler::opMPIWaitAllIfStmtMatcher,
                    &seqKernelHandler);
  Finder.addMatcher(SeqKernelHandler::mapIdxDeclMatcher, &seqKernelHandler);
}

} // namespace OP2
