#include "OMPKernelHandler.h"
#include "generators/common/ASTMatchersExtension.h"
#include "generators/common/handler.hpp"
#include "clang/AST/StmtOpenMP.h"

namespace {
using namespace clang::ast_matchers;
const auto parLoopSkeletonCompStmtMatcher =
    compoundStmt(hasParent(functionDecl(hasName("op_par_loop_skeleton"))));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;
//___________________________________MATCHERS__________________________________

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
    ompParallelForDirective().bind(
        "ompParForDir"); // FIXME check if it is in the main file.

//_________________________________CONSTRUCTORS________________________________
OMPKernelHandler::OMPKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

//________________________________GLOBAL_HANDLER_______________________________
void OMPKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!lineReplHandler<DeclStmt, 1>(
          Result, Replace, "local_reduction_variable",
          std::bind(&OMPKernelHandler::handleRedLocalVarDecl, this)))
    return;
  if (!HANDLER(CallExpr, 2, "func_call", OMPKernelHandler::handleFuncCall))
    return; // if successfully handled return
  if (!lineReplHandler<BinaryOperator, 7>(
          Result, Replace, "loc_red_to_arg_assignment",
          std::bind(&OMPKernelHandler::handlelocRedToArgAssignment, this)))
    return; // if successfully handled return
  if (!HANDLER(OMPParallelForDirective, 0, "ompParForDir",
               OMPKernelHandler::handleOMPParLoop)) {
    return;
  }
}
//___________________________________HANDLERS__________________________________

std::string OMPKernelHandler::handleFuncCall() {
  std::string funcCall = "";
  llvm::raw_string_ostream ss(funcCall);
  ss << loop.getUserFuncInfo().funcName << "(";
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (!loop.getArg(i).isReduction()) {
      if (loop.getArg(i).isDirect()) {
        ss << loop.getArg(i).getArgCall(i, "n");
      } else {

        ss << loop.getArg(i).getArgCall(
            loop.dat2argIdxs[loop.dataIdxs[i]],
            ("map" + std::to_string(loop.mapIdxs[i]) + "idx"));
      }
      ss << ",";
    } else {
      ss << "&arg" << i << "_l,";
    }
  }
  ss.str();
  return funcCall.substr(0, funcCall.length() - 1) + ");";
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

std::string OMPKernelHandler::handleOMPParLoop() {
  std::string plusReds, minReds, maxReds;
  llvm::raw_string_ostream os(plusReds);
  llvm::raw_string_ostream osMin(minReds);
  llvm::raw_string_ostream osMax(maxReds);
  for (size_t ind = 0; ind < loop.getNumArgs(); ++ind) {
    const OPArg &arg = loop.getArg(ind);
    if (arg.isReduction()) {
      switch (arg.accs) {
      case OP2::OP_INC:
        os << "arg" << ind << "_l, ";
        break;
      case OP2::OP_MAX:
        osMax << "arg" << ind << "_l, ";
        break;
      case OP2::OP_MIN:
        osMin << "arg" << ind << "_l, ";
        break;
      default:
        // error if this is a reduction it must be one of OP_MIN, OP_MAX or
        // OP_INC
        assert(!arg.isReduction() ||
               (arg.accs == OP2::OP_INC || arg.accs == OP2::OP_MAX ||
                arg.accs == OP2::OP_MIN));
      }
    }
  }
  if (os.str().length() > 0) {
    plusReds =
        " reduction(+:" + plusReds.substr(0, plusReds.length() - 2) + ")";
  }
  if (osMin.str().length() > 0) {
    minReds = " reduction(min:" + minReds.substr(0, minReds.length() - 2) + ")";
  }
  if (osMax.str().length() > 0) {
    maxReds = " reduction(max:" + maxReds.substr(0, maxReds.length() - 2) + ")";
  }
  return "omp parallel for " + plusReds + " " + minReds + " " + maxReds;
}

} // namespace OP2
