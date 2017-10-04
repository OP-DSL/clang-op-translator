#include "BaseKernelHandler.h"
#include "../utils.h"

namespace OP2 {
using namespace clang::ast_matchers;

const DeclarationMatcher BaseKernelHandler::parLoopDeclMatcher =
    functionDecl(hasName("op_par_loop_skeleton")).bind("par_loop_decl");
const DeclarationMatcher BaseKernelHandler::nargsMatcher =
    varDecl(hasType(isInteger()), hasName("nargs")).bind("nargs_decl");

const DeclarationMatcher BaseKernelHandler::argsArrMatcher =
    varDecl(hasName("args"),
            hasParent(declStmt(hasParent(compoundStmt(
                hasParent(functionDecl(hasName("op_par_loop_skeleton"))))))))
        .bind("args_arr_decl");
const StatementMatcher BaseKernelHandler::argsArrSetterMatcher =
    cxxOperatorCallExpr(/*hasType(cxxRecordDecl(hasName("op_arg"))),FIXME more
                           specific matcher*/
                        hasParent(compoundStmt(hasParent(
                            functionDecl(hasName("op_par_loop_skeleton"))))))
        .bind("args_element_setter");
const StatementMatcher BaseKernelHandler::opTimingReallocMatcher =
    callExpr(callee((functionDecl(hasName("op_timing_realloc")))),
             hasParent(compoundStmt(
                 hasParent(functionDecl(hasName("op_par_loop_skeleton"))))))
        .bind("op_timing_realloc");

BaseKernelHandler::BaseKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

void BaseKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!handleParLoopDecl(Result))
    return; // if successfully handled return
  if (!handleNargsDecl(Result))
    return;
  if (!handleArgsArrDecl(Result))
    return;
  if (!handleArgsArrSetter(Result))
    return;
  if (!handleOPTimingRealloc(Result))
    return;
}

int BaseKernelHandler::handleParLoopDecl(
    const MatchFinder::MatchResult &Result) {
  const clang::FunctionDecl *function =
      Result.Nodes.getNodeAs<clang::FunctionDecl>("par_loop_decl");

  if (!function)
    return 1; // We shouldn't handle this match
  clang::SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(function->getLocStart(), sm);
  // replace skeleton to the name of the loop
  size_t nameoffset = std::string("void op_par_loop_").length();
  size_t length = std::string("skeleton").length();
  clang::tooling::Replacement funcNameRep(
      *sm, function->getLocStart().getLocWithOffset(nameoffset), length,
      loop.getName());
  if (llvm::Error err = (*Replace)[filename].add(funcNameRep)) {
    // TODO diagnostics..
    llvm::errs() << "Function name replacement failed in: " << filename << "\n";
  }
  // add op_args to the parameter list
  std::string arg_repl = "";
  llvm::raw_string_ostream os(arg_repl);
  for (size_t i = 1; i < loop.getNumArgs(); ++i) {
    os << ", op_arg arg" << i;
  }

  if (loop.getNumArgs() > 1) {
    clang::tooling::Replacement funcArgRep(
        *sm,
        function->getParamDecl(function->getNumParams() - 1)
            ->getLocEnd()
            .getLocWithOffset(4 /*FIXME hardcoded length of arg0*/),
        0, os.str());
    if (llvm::Error err = (*Replace)[filename].add(funcArgRep)) {
      // TODO diagnostics..
      llvm::errs() << "Function arg addition failed in: " << filename << "\n";
    }
  }
  return 0;
}

int BaseKernelHandler::handleNargsDecl(const MatchFinder::MatchResult &Result) {
  const clang::VarDecl *nargsDecl =
      Result.Nodes.getNodeAs<clang::VarDecl>("nargs_decl");
  if (!nargsDecl)
    return 1; // We shouldn't handle this match
  clang::SourceManager *sm = Result.SourceManager;
  if (!sm->isWrittenInMainFile(nargsDecl->getLocStart()))
    return 0;
  std::string filename = getFileNameFromSourceLoc(nargsDecl->getLocStart(), sm);

  // change value of nargs
  if (loop.getNumArgs() <= 1)
    return 0; // there is no need to change
  clang::tooling::Replacement repl(*sm, nargsDecl->getLocEnd(),
                                   1 /*FIXME hardcoded len of 1..*/,
                                   std::to_string(loop.getNumArgs()));
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set value of nargs failed in: " << filename << "\n";
  }

  return 0;
}

int BaseKernelHandler::handleArgsArrDecl(
    const MatchFinder::MatchResult &Result) {
  const clang::VarDecl *argsArrDecl =
      Result.Nodes.getNodeAs<clang::VarDecl>("args_arr_decl");

  if (!argsArrDecl)
    return 1; // We shouldn't handle this match
  clang::SourceManager *sm = Result.SourceManager;
  if (!sm->isWrittenInMainFile(argsArrDecl->getLocStart()))
    return 0;
  std::string filename =
      getFileNameFromSourceLoc(argsArrDecl->getLocStart(), sm);

  // change value of array size
  if (loop.getNumArgs() <= 1)
    return 0; // there is no need to change

  clang::tooling::Replacement repl(*sm,
                                   argsArrDecl->getLocEnd().getLocWithOffset(
                                       -1) /*FIXME hardcoded len of 1..*/,
                                   1 /*FIXME hardcoded len of 1..*/,
                                   std::to_string(loop.getNumArgs()));
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set value of nargs failed in: " << filename << "\n";
  }

  return 0;
}

int BaseKernelHandler::handleArgsArrSetter(
    const MatchFinder::MatchResult &Result) {
  const clang::CXXOperatorCallExpr *argsElementSetterExpr =
      Result.Nodes.getNodeAs<clang::CXXOperatorCallExpr>("args_element_setter");
  if (!argsElementSetterExpr)
    return 1; // We shouldn't handle this match
  clang::SourceManager *sm = Result.SourceManager;

  if (!sm->isWrittenInMainFile(argsElementSetterExpr->getLocStart()))
    return 0;
  std::string filename =
      getFileNameFromSourceLoc(argsElementSetterExpr->getLocStart(), sm);

  std::string replacement = "";
  llvm::raw_string_ostream os(replacement);
  for (unsigned i = 0; i < loop.getNumArgs(); ++i) {
    os << "args[" << i << "] = arg" << i << ";\n";
  }
  clang::SourceRange replRange(
      argsElementSetterExpr->getLocStart(),
      argsElementSetterExpr->getLocEnd().getLocWithOffset(
          5 /*FIXME hardcoded len 'arg0;'*/));
  clang::tooling::Replacement repl(
      *sm, clang::CharSourceRange(replRange, false), os.str());
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set value of nargs failed in: " << filename << "\n";
  }

  return 0;
}

int BaseKernelHandler::handleOPTimingRealloc(
    const matchers::MatchFinder::MatchResult &Result) {
  const clang::CallExpr *opTimingReallocCallExpr =
      Result.Nodes.getNodeAs<clang::CallExpr>("op_timing_realloc");
  if (!opTimingReallocCallExpr)
    return 1; // We shouldn't handle this match

  clang::SourceManager *sm = Result.SourceManager;

  if (!sm->isWrittenInMainFile(opTimingReallocCallExpr->getLocStart()))
    return 0;
  std::string filename =
      getFileNameFromSourceLoc(opTimingReallocCallExpr->getLocStart(), sm);

  clang::SourceRange replRange(
      opTimingReallocCallExpr->getArg(0)->getLocStart(),
      opTimingReallocCallExpr->getArg(0)->getLocEnd().getLocWithOffset(
          1 /*FIXME hardcoded len 0*/));
  clang::tooling::Replacement repl(*sm,
                                   clang::CharSourceRange(replRange, false),
                                   "0/*changed*/" /*FIXME hardcoded loopID*/);
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set value of nargs failed in: " << filename << "\n";
  }

  return 0;
}

} // end of namespace OP2
