#include "SeqKernelHandler.h"
#include "../utils.h"

namespace {
using namespace clang::ast_matchers;
const auto parLoopSkeletonCompStmtMatcher =
    compoundStmt(hasParent(functionDecl(hasName("op_par_loop_skeleton"))));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;

const DeclarationMatcher SeqKernelHandler::userFuncMatcher =
    functionDecl(hasName("skeleton"), isDefinition(), parameterCountIs(1))
        .bind("user_func");
const StatementMatcher SeqKernelHandler::funcCallMatcher =
    callExpr(callee(functionDecl(hasName("skeleton"), parameterCountIs(1))))
        .bind("func_call");
const StatementMatcher SeqKernelHandler::opMPIReduceMatcher =
    callExpr(
        callee(functionDecl(hasName("op_mpi_reduce"), parameterCountIs(2))),
        hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("reduce_func_call");
const StatementMatcher SeqKernelHandler::opMPIWaitAllIfStmtMatcher =
    ifStmt(hasThen(compoundStmt(statementCountIs(1),
                                hasAnySubstatement(callExpr(callee(functionDecl(
                                    hasName("op_mpi_wait_all"))))))))
        .bind("wait_all_if");
const DeclarationMatcher SeqKernelHandler::mapIdxDeclMatcher =
    varDecl(
        hasName(
            "map0idx") /*,TODO more specific matcher? hasParent(compoundStmt(hasParent(forStmt(hasParent(compoundStmt(hasParent(ifStmt(hasParent(parLoopSkeletonCompStmtMatcher)))))))))*/)
        .bind("map_idx_decl");

SeqKernelHandler::SeqKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

void SeqKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!handleUserFuncDecl(Result))
    return; // if successfully handled return
  if (!handleUserFuncCall(Result))
    return; // if successfully handled return
  if (!handleMPIReduceCall(Result))
    return; // if successfully handled return
  if (!handleMPIWaitAllIfStmt(Result))
    return; // if successfully handled return
  if (!handleMapIdxDecl(Result))
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

int SeqKernelHandler::handleMPIReduceCall(
    const MatchFinder::MatchResult &Result) {
  const CallExpr *funcCall =
      Result.Nodes.getNodeAs<CallExpr>("reduce_func_call");
  if (!funcCall)
    return 1;
  SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(funcCall->getLocStart(), sm);
  SourceRange replRange(funcCall->getLocStart(),
                        funcCall->getLocEnd().getLocWithOffset(2));
  /*FIXME magic number for semicolon pos*/

  tooling::Replacement repl(*sm, CharSourceRange(replRange, false),
                            loop.getMPIReduceCall());

  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement of op_mpi_reduce call in: " << filename
                 << "\n";
  }
  return 0;
}

int SeqKernelHandler::handleMPIWaitAllIfStmt(
    const MatchFinder::MatchResult &Result) {
  const IfStmt *ifStmt = Result.Nodes.getNodeAs<IfStmt>("wait_all_if");
  if (!ifStmt)
    return 1;
  if (!loop.isDirect())
    return 0;

  SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(ifStmt->getLocStart(), sm);
  SourceRange replRange(ifStmt->getLocStart(),
                        ifStmt->getLocEnd().getLocWithOffset(1));
  /*FIXME magic number for semicolon pos*/

  tooling::Replacement repl(*sm, CharSourceRange(replRange, false), "");

  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement of op_mpi_wat_all failed in: " << filename
                 << "\n";
  }
  return 0;
}

int SeqKernelHandler::handleMapIdxDecl(const MatchFinder::MatchResult &Result) {
  const VarDecl *mapIdx = Result.Nodes.getNodeAs<VarDecl>("map_idx_decl");
  if (!mapIdx)
    return 1;

  SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(mapIdx->getLocStart(), sm);
  SourceRange replRange(mapIdx->getLocStart(),
                        mapIdx->getLocEnd().getLocWithOffset(2));
  /*FIXME magic number for semicolon pos*/

  tooling::Replacement repl(*sm, CharSourceRange(replRange, false),
                            loop.getMapVarDecls());

  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement of op_mpi_wat_all failed in: " << filename
                 << "\n";
  }
  return 0;
}

} // namespace OP2
