#include "OMPKernelHandler.h"
#include "../utils.h"

namespace {
using namespace clang::ast_matchers;
const auto parLoopSkeletonCompStmtMatcher =
    compoundStmt(hasParent(functionDecl(hasName("op_par_loop_skeleton"))));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;
///__________________________________MATCHERS__________________________________

const StatementMatcher OMPKernelHandler::locRedVarMatcher =
    declStmt(containsDeclaration(0, varDecl(hasName("arg0h"))),
             hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("local_reduction_variable");

///________________________________CONSTRUCTORS________________________________
OMPKernelHandler::OMPKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

///_______________________________GLOBAL_HANDLER_______________________________
void OMPKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!handleRedLocalVarDecl(Result))
    return; // if successfully handled return
}
///__________________________________HANDLERS__________________________________

int OMPKernelHandler::handleRedLocalVarDecl(
    const MatchFinder::MatchResult &Result) {
  const DeclStmt *varDecl =
      Result.Nodes.getNodeAs<DeclStmt>("local_reduction_variable");
  if (!varDecl)
    return 1;
  SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(varDecl->getLocStart(), sm);
  SourceRange replRange(varDecl->getLocStart(),
                        varDecl->getLocEnd().getLocWithOffset(1));
  tooling::Replacement repl(*sm, CharSourceRange(replRange, false),
                            "" /*TODO*/);
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement of local reduction variables failed in: "
                 << filename << "\n";
  }
  return 0;
}
} // namespace OP2
