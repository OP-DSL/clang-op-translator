#include <llvm/Support/Debug.h>
#include "ParLoopHandler.h"
#include "utils.h"

namespace OP2 {

void ParLoopHandler::run(const matchers::MatchFinder::MatchResult &Result) {
  const clang::CallExpr* function = Result.Nodes.getNodeAs<clang::CallExpr>("par_loop");
  const clang::Expr* str_arg = function->getArg(1);
  const clang::StringLiteral* name = getAsStringLiteral(str_arg);

  // If the second argument isn't a string literal, issue an error
  if (!name) {
    reportDiagnostic(*Result.Context, str_arg, "op_par_loop called with non-string literal kernel name");
  }
  debugs() << "processing kernel " << name->getString() << " with " << function->getNumArgs() << " arguments\n";

  llvm::iterator_range<clang::CallExpr::const_arg_iterator> args {function->arg_begin()+2, function->arg_end()};

  // Iterate over arguments
  for (auto& arg : args) {
      debugs() << arg->getType().getAsString() << ",";
  }
  debugs() << "\n";
}
}
