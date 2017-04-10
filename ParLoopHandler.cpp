#include "ParLoopHandler.h"
#include "utils.h"

namespace OP2 {

void ParLoopHandler::run(const matchers::MatchFinder::MatchResult &Result) {
  const clang::CallExpr* function = Result.Nodes.getNodeAs<clang::CallExpr>("par_loop");
  const clang::Expr* str_arg = function->getArg(1);
  const clang::StringLiteral* name = getAsStringLiteral(str_arg);
  if (!name) {
    reportDiagnostic(*Result.Context, str_arg, "op_par_loop called with non-string literal kernel name");
  }
  llvm::outs() << "processing kernel " << name->getString() << " with " << function->getNumArgs() << " arguments\n";
}
}
