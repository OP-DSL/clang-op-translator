#include "SeqKernelHandler.h"
#include "../utils.h"

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;

const DeclarationMatcher SeqKernelHandler::userFuncMatcher =
    functionDecl(hasName("skeleton"), isDefinition(), parameterCountIs(1))
        .bind("user_func");
const StatementMatcher SeqKernelHandler::funcCallMatcher =
    callExpr(callee(functionDecl(hasName("skeleton"), parameterCountIs(1))))
        .bind("func_call");

SeqKernelHandler::SeqKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

void SeqKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!handleUserFuncDecl(Result))
    return; // if successfully handled return
  if (!handleUserFuncCall(Result))
    return; // if successfully handled return
}

int SeqKernelHandler::handleUserFuncDecl(
    const MatchFinder::MatchResult &Result) {
  const FunctionDecl *userFunc =
      Result.Nodes.getNodeAs<FunctionDecl>("user_func");
  if (!userFunc)
    return 1;
  SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(userFunc->getLocStart(), sm);
  SourceRange replRange(userFunc->getLocStart(),
                        userFunc->getLocEnd().getLocWithOffset(1));
  tooling::Replacement repl(*sm, CharSourceRange(replRange, false),
                            loop.getUserFuncInc());
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement of user function failed in: " << filename
                 << "\n";
  }
  return 0;
}

int SeqKernelHandler::handleUserFuncCall(
    const MatchFinder::MatchResult &Result) {
  const CallExpr *funcCall = Result.Nodes.getNodeAs<CallExpr>("func_call");
  if (!funcCall)
    return 1;
  SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(funcCall->getLocStart(), sm);
  SourceRange replRange(funcCall->getLocStart(),
                        funcCall->getLocEnd().getLocWithOffset(2));
  /*FIXME magic number for semicolon pos*/

  tooling::Replacement repl(*sm, CharSourceRange(replRange, false),
                            loop.getFuncCall());
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement of user function call in: " << filename
                 << "\n";
  }
  return 0;
}

} // namespace OP2
