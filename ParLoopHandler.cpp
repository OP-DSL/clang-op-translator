#include "ParLoopHandler.h"
#include "utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <llvm/Support/Debug.h>
#include <sstream>

namespace OP2 {

void ParLoopHandler::run(const matchers::MatchFinder::MatchResult &Result) {
  const clang::CallExpr *function =
      Result.Nodes.getNodeAs<clang::CallExpr>("par_loop");
  if (function->getNumArgs() < 3) {
    reportDiagnostic(*Result.Context, function,
                     "not enough arguments to op_par_loop");
  }
  const clang::Expr *str_arg = function->getArg(1);
  const clang::StringLiteral *name = getAsStringLiteral(str_arg);

  // If the second argument isn't a string literal, issue an error
  if (!name) {
    reportDiagnostic(*Result.Context, str_arg,
                     "op_par_loop called with non-string literal kernel name",
                     clang::DiagnosticsEngine::Warning);
  }
  debugs() << "processing kernel " << name->getString() << " with "
           << function->getNumArgs() << " arguments\n";

  const clang::FunctionDecl *parent =
      findParent<clang::FunctionDecl>(*function, *Result.Context);
  std::stringstream ss;
  ss << "// op_par_loop kernel " << name->getString().str()
     << " definition here\n\n";
  Rewriter.InsertTextBefore(parent->getLocStart(), ss.str());

  // Iterate over arguments
  for (auto *arg :
       llvm::make_range(function->arg_begin() + 3, function->arg_end())) {

    auto arg_gbl_processor =
        make_matcher([](const matchers::MatchFinder::MatchResult &Result) {});

    auto find_function_call = [](llvm::StringRef s) {
      using namespace clang::ast_matchers;
      return stmt(
          hasDescendant(callExpr(callee(functionDecl(hasName(s)))).bind(s)));
    };

    clang::ast_matchers::MatchFinder Matcher;
    // arg->dump();
    using namespace std::placeholders;
    auto arg_dat_match_processor =
        make_matcher(std::bind(&ParLoopHandler::arg_dat_processor, this, _1));
    auto arg_gbl_match_processor =
        make_matcher(std::bind(&ParLoopHandler::arg_gbl_processor, this, _1));
    Matcher.addMatcher(find_function_call("op_arg_dat"), &arg_dat_match_processor);

    Matcher.addMatcher(find_function_call("op_arg_gbl"), &arg_gbl_match_processor);

    auto errorMatcher = make_matcher([arg](
        const matchers::MatchFinder::MatchResult &Result) {
      reportDiagnostic(
          *Result.Context, arg,
          "argument to op_par_loop must be a call to op_arg_dat or op_arg_gbl");
    });

    // Error if the argument isn't a call to op_arg_dat or op_arg_gbl.
    using namespace clang::ast_matchers;
    Matcher.addMatcher(
        stmt(unless(hasDescendant(callExpr(callee(functionDecl(
            anyOf(hasName("op_arg_dat"), hasName("op_arg_gbl")))))))),
        &errorMatcher);
    Matcher.match(*arg, *Result.Context);
  }
}

void ParLoopHandler::arg_dat_processor(const matchers::MatchFinder::MatchResult &Result) {
  const clang::CallExpr *call =
    Result.Nodes.getNodeAs<clang::CallExpr>("op_arg_dat");
}

void ParLoopHandler::arg_gbl_processor(
    const matchers::MatchFinder::MatchResult &Result) {}
}
