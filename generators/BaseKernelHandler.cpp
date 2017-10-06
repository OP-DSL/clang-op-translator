#include "BaseKernelHandler.h"
#include "../utils.h"

namespace {
using namespace clang::ast_matchers;
const auto parLoopSkeletonCompStmtMatcher =
    compoundStmt(hasParent(functionDecl(hasName("op_par_loop_skeleton"))));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
// Static Matchers of BaseKernelHandler
const DeclarationMatcher BaseKernelHandler::parLoopDeclMatcher =
    functionDecl(hasName("op_par_loop_skeleton")).bind("par_loop_decl");
const DeclarationMatcher BaseKernelHandler::nargsMatcher =
    varDecl(hasType(isInteger()), hasName("nargs")).bind("nargs_decl");

const DeclarationMatcher BaseKernelHandler::argsArrMatcher =
    varDecl(hasName("args"),
            hasParent(declStmt(hasParent(parLoopSkeletonCompStmtMatcher))))
        .bind("args_arr_decl");
const StatementMatcher BaseKernelHandler::argsArrSetterMatcher =
    cxxOperatorCallExpr(/*hasType(cxxRecordDecl(hasName("op_arg"))),FIXME more
                           specific matcher*/
                        hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("args_element_setter");
const StatementMatcher BaseKernelHandler::opTimingReallocMatcher =
    callExpr(callee(functionDecl(hasName("op_timing_realloc"))),
             hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("op_timing_realloc");
const StatementMatcher BaseKernelHandler::printfKernelNameMatcher =
    callExpr(callee(functionDecl(hasName("printf"))),
             hasParent(compoundStmt(
                 hasParent(ifStmt(hasParent(parLoopSkeletonCompStmtMatcher))))))
        .bind("printfName"); // More spec needed
const StatementMatcher BaseKernelHandler::opKernelsSubscriptMatcher =
    arraySubscriptExpr(hasBase(implicitCastExpr(hasSourceExpression(
                           declRefExpr(to(varDecl(hasName("OP_kernels"))))))),
                       hasIndex(integerLiteral(equals(0))),
                       hasParent(memberExpr(hasParent(binaryOperator().bind(
                                                "op_kernels_assignment")))
                                     .bind("opk_member_expr")))
        .bind("op_kernels_index");

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
  if (!handleOPDiagPrintf(Result))
    return;
  if (!handleOPKernels(Result))
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
    llvm::errs() << "Set size for args array failed in: " << filename << "\n";
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
    llvm::errs() << "Set args array failed in: " << filename << "\n";
  }

  return 0;
}

int BaseKernelHandler::handleOPTimingRealloc(
    const MatchFinder::MatchResult &Result) {
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
                                   std::to_string(loop.getLoopID()));
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set loopID for timing_realloc failed at: " << filename
                 << "\n";
  }

  return 0;
}

int BaseKernelHandler::handleOPDiagPrintf(
    const MatchFinder::MatchResult &Result) {
  const clang::CallExpr *printfCallExpr =
      Result.Nodes.getNodeAs<clang::CallExpr>("printfName");
  if (!printfCallExpr)
    return 1; // We shouldn't handle this match

  clang::SourceManager *sm = Result.SourceManager;
  if (!sm->isWrittenInMainFile(printfCallExpr->getLocStart()))
    return 0;
  std::string filename =
      getFileNameFromSourceLoc(printfCallExpr->getLocStart(), sm);

  clang::SourceRange replRange(
      printfCallExpr->getArg(0)->getLocEnd(),
      printfCallExpr->getArg(0)->getLocEnd().getLocWithOffset(
          2 /*FIXME hardcoded*/));
  std::string replString = std::string("\" kernel routine ") +
                           (loop.isDirect() ? "w/o" : "with") +
                           " indirection:  " + loop.getName() + "\"";
  clang::tooling::Replacement repl(
      *sm, clang::CharSourceRange(replRange, false), replString);
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set printf failed in: " << filename << "\n";
  }

  return 0;
}

int BaseKernelHandler::handleOPKernels(const MatchFinder::MatchResult &Result) {
  const clang::ArraySubscriptExpr *kernelsSubscriptExpr =
      Result.Nodes.getNodeAs<clang::ArraySubscriptExpr>("op_kernels_index");
  if (!kernelsSubscriptExpr)
    return 1; // We shouldn't handle this match

  clang::SourceManager *sm = Result.SourceManager;
  if (!sm->isWrittenInMainFile(kernelsSubscriptExpr->getLocStart()))
    return 0;
  std::string filename =
      getFileNameFromSourceLoc(kernelsSubscriptExpr->getLocStart(), sm);

  if (Result.Nodes.getNodeAs<clang::MemberExpr>("opk_member_expr")
          ->getMemberDecl()
          ->getNameAsString() == "transfer") {
    const clang::BinaryOperator *bop =
        Result.Nodes.getNodeAs<clang::BinaryOperator>("op_kernels_assignment");
    /*TODO generate proper transfer data*/
    clang::SourceRange replRange(bop->getLocStart(),
                                 bop->getLocEnd().getLocWithOffset(4));
    // clang::arcmt::trans::findSemiAfterLocation(bop->getLocEnd(),
    // *Result.Context));
    clang::tooling::Replacement repl(
        *sm, clang::CharSourceRange(replRange, false), loop.getTransferData());
    if (llvm::Error err = (*Replace)[filename].add(repl)) {
      // TODO diagnostics..
      llvm::errs() << "Set transfer failed in: " << filename << "\n";
    }
    return 0;
  }
  clang::tooling::Replacement repl(
      *sm, kernelsSubscriptExpr->getIdx()->getLocStart(), 1,
      std::to_string(loop.getLoopID()));
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set looID in for index OP_kernels failed in: " << filename
                 << "\n";
  }

  return 0;
}
} // end of namespace OP2
