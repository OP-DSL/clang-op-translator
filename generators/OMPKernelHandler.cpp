#include "OMPKernelHandler.h"
#include "handler.hpp"
#include "clang/AST/StmtOpenMP.h"

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
    declStmt(containsDeclaration(0, varDecl(hasName("arg0_l"))),
             hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("local_reduction_variable");

const StatementMatcher OMPKernelHandler::locRedToArgMatcher =
    binaryOperator(
        hasOperatorName("="),
        hasRHS(ignoringImpCasts(declRefExpr(to(varDecl(hasName("arg0_l")))))),
        hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("loc_red_to_arg_assignment");
const StatementMatcher OMPKernelHandler::ompParForMatcher =
    ifStmt(hasCondition(binaryOperator(hasRHS(integerLiteral(equals(0))))),
           hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("if_omp_par_loop"); // FIXME matcher for the pragma..

///________________________________CONSTRUCTORS________________________________
OMPKernelHandler::OMPKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

///_______________________________GLOBAL_HANDLER_______________________________
void OMPKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!lineReplHandler<DeclStmt, 1>(
          Result, Replace, "local_reduction_variable",
          std::bind(&OMPKernelHandler::handleRedLocalVarDecl, this)))
    return;
  if (!handleOMPParLoop(Result))
    return; // if successfully handled return
  if (!HANDLER(CallExpr, 2, "func_call", OMPKernelHandler::handleFuncCall))
    return; // if successfully handled return
  if (!lineReplHandler<BinaryOperator, 7>(
          Result, Replace, "loc_red_to_arg_assignment",
          std::bind(&OMPKernelHandler::handlelocRedToArgAssignment, this)))
    return; // if successfully handled return
}
///__________________________________HANDLERS__________________________________

std::string OMPKernelHandler::handleFuncCall() {
  std::string funcCall = "";
  llvm::raw_string_ostream ss(funcCall);
  ss << loop.getName() << "("; // TODO fix repr to store correct function data.
  for (size_t i = 0; i < loop.getNumArgs() - 1; ++i) {
    if (!loop.getArg(i).isReduction()) {
      ss << loop.getArg(i).getArgCall(i,
                                      loop.getArg(i).isDirect()
                                          ? "n"
                                          : ("map" + std::to_string(i) + "idx"))
         << ",\n";
    } else {
      ss << "&arg" << i << "_l, ";
    }
  }
  size_t n = loop.getNumArgs() - 1;
  if (!loop.getArg(n).isReduction()) {
    ss << loop.getArg(n).getArgCall(
        n,
        loop.getArg(n).isDirect() ? "n" : ("map" + std::to_string(n) + "idx"));
  } else {
    ss << "arg" << n << "_l";
  }
  ss << "\n);";
  return ss.str();
}

std::string OMPKernelHandler::handleRedLocalVarDecl() {
  std::string s;
  llvm::raw_string_ostream os(s);
  for (size_t ind = 0; ind < loop.getNumArgs(); ++ind) {
    const OPArg &arg = loop.getArg(ind);
    if (arg.isReduction()) {
      os << arg.type << " arg" << ind << "_l = *(" + arg.type + " *)arg" << ind
         << ".data;\n";
    }
  }
  return os.str();
}

std::string OMPKernelHandler::handlelocRedToArgAssignment() {
  std::string s;
  llvm::raw_string_ostream os(s);
  for (size_t ind = 0; ind < loop.getNumArgs(); ++ind) {
    const OPArg &arg = loop.getArg(ind);
    if (arg.isReduction()) {
      os << "*((" + arg.type + " *)arg" << ind << ".data) = arg" << ind
         << "_l;\n";
    }
  }
  return os.str();
}

int OMPKernelHandler::handleOMPParLoop(
    const matchers::MatchFinder::MatchResult &Result) {
  const IfStmt *match = Result.Nodes.getNodeAs<IfStmt>("if_omp_par_loop");
  if (!match)
    return 1;
  SourceManager *sm = Result.SourceManager;
  if (!sm->isWrittenInMainFile(match->getLocStart()))
    return 0;
  std::string filename = getFileNameFromSourceLoc(match->getLocStart(), sm);
  const OMPParallelForDirective *directive =
      llvm::dyn_cast<OMPParallelForDirective>(
          *(match->getThen()->child_begin()));

  SourceRange replRange(directive->getLocStart(), directive->getLocEnd());

  std::string s;
  llvm::raw_string_ostream os(s);
  for (size_t ind = 0; ind < loop.getNumArgs(); ++ind) {
    const OPArg &arg = loop.getArg(ind);
    if (arg.isReduction()) { // FIXME min max reductions
      os << "arg" << ind << "_l, ";
    }
  }
  if (os.str().length() > 0) {
    s = " reduction(+:" + s.substr(0, s.length() - 2) + ")";
  }
  s = "omp parallel for " + s;
  tooling::Replacement repl(*sm, CharSourceRange(replRange, false), s);
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement of omp directive failed in: " << filename
                 << "\n";
  }
  return 0;
}
} // namespace OP2
