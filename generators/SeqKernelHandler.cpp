#include "SeqKernelHandler.h"
#include "../utils.h"
#include "handler.hpp"

namespace {
using namespace clang::ast_matchers;
const auto parLoopSkeletonCompStmtMatcher =
    compoundStmt(hasParent(functionDecl(hasName("op_par_loop_skeleton"))));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;

///__________________________________MATCHERS__________________________________
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
    varDecl(hasName("map0idx"), hasAncestor(parLoopSkeletonCompStmtMatcher))
        .bind("map_idx_decl");

///________________________________CONSTRUCTORS________________________________
SeqKernelHandler::SeqKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

///_______________________________GLOBAL_HANDLER_______________________________
void SeqKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!lineReplHandler<FunctionDecl, 1>(Result, Replace, "user_func", [this]() {
        return this->loop.getUserFuncInc();
      }))
    return; // if successfully handled return
  if (!lineReplHandler<FunctionDecl, 2>(Result, Replace, "func_call", [this]() {
        return this->loop.getFuncCall();
      }))
    return; // if successfully handled return
  if (!lineReplHandler<FunctionDecl, 2>(
          Result, Replace, "reduce_func_call",
          [this]() { return this->loop.getMPIReduceCall(); }))
    return; // if successfully handled return
  if (!handleMPIWaitAllIfStmt(Result))
    return; // if successfully handled return
  if (!lineReplHandler<FunctionDecl, 2>(
          Result, Replace, "map_idx_decl",
          [this]() { return this->loop.getMapVarDecls(); }))
    return; // if successfully handled return
}

///__________________________________HANDLERS__________________________________
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

} // namespace OP2
