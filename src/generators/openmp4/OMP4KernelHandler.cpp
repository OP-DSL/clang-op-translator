#include "OMP4KernelHandler.h"
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

const StatementMatcher OMP4KernelHandler::locRedVarMatcher =
    declStmt(containsDeclaration(0, varDecl(hasName("arg0_l"))),
             hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("local_reduction_variable");

const StatementMatcher OMP4KernelHandler::locRedToArgMatcher =
    binaryOperator(
        hasOperatorName("="),
        hasRHS(ignoringImpCasts(declRefExpr(to(varDecl(hasName("arg0_l")))))),
        hasParent(parLoopSkeletonCompStmtMatcher))
        .bind("loc_red_to_arg_assignment");

const StatementMatcher OMP4KernelHandler::ompParForMatcher =
    ompParallelForDirective().bind(
        "ompParForDir"); // FIXME check if it is in the main file.

const DeclarationMatcher OMP4KernelHandler::userFuncMatcher =
    functionDecl(hasName("skeleton_OMP4"), isDefinition(), parameterCountIs(1))
        .bind("user_func_OMP4");

const StatementMatcher OMP4KernelHandler::funcCallMatcher =
    callExpr(callee(functionDecl(hasName("skeleton_OMP4"), parameterCountIs(1))))
        .bind("func_call_OMP4");

        
const DeclarationMatcher OMP4KernelHandler::mapIdxDeclMatcher =
    varDecl(hasName("map_"), hasAncestor(parLoopSkeletonCompStmtMatcher))
        .bind("map_decl");


//_________________________________CONSTRUCTORS________________________________
OMP4KernelHandler::OMP4KernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop, const OP2Application &application, const size_t loopIdx)
    : Replace(Replace), loop(loop), application(application), loopIdx(loopIdx) {}

//________________________________GLOBAL_HANDLER_______________________________
void OMP4KernelHandler::run(const MatchFinder::MatchResult &Result) {

  if (!lineReplHandler<FunctionDecl, 1>(Result, Replace, "user_func_OMP4",  [this]() {
        return this->getmappedFunc();
      }))
    return; // if successfully handled return

  if (!lineReplHandler<VarDecl, 1>(Result, Replace, "map_decl",  [this]() {
        return this->DevicePointerDecl();
      }))
    return; // if successfully handled return

  if (!lineReplHandler<DeclStmt, 1>(
          Result, Replace, "local_reduction_variable",
          std::bind(&OMP4KernelHandler::handleRedLocalVarDecl, this)))
    return;
  if (!HANDLER(CallExpr, 2, "func_call", OMP4KernelHandler::handleFuncCall))
    return; // if successfully handled return
  if (!lineReplHandler<BinaryOperator, 7>(
          Result, Replace, "loc_red_to_arg_assignment",
          std::bind(&OMP4KernelHandler::handlelocRedToArgAssignment, this)))
    return; // if successfully handled return
  if (!HANDLER(OMPParallelForDirective, 0, "ompParForDir",
               OMP4KernelHandler::handleOMPParLoop)) {
    return;
  }
}
//___________________________________HANDLERS__________________________________

std::string OMP4KernelHandler::handleFuncCall() {
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

std::string OMP4KernelHandler::getmappedFunc(){
  std::string mappedfunc = "";
  llvm::raw_string_ostream ss(mappedfunc);
  std::map<std::string,std::string> arg2data;
  ss << "void " << loop.getName() << "_omp4_kernel(";
  if(!loop.isDirect()){
    ss << "int *map0, int map0size,";
  }
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if(arg2data[loop.getArg(i).opDat] != ""){
      continue;
    } else {
      arg2data[loop.getArg(i).opDat] = "data" + std::to_string(i);
    }
    ss << loop.getArg(i).type << " *" << "data" << i << ", ";
    ss << "int " << "data" << i << "size ";
    if(i != loop.getNumArgs() -1 ){
      ss << ",";
    }
  }
  ss << " int count, int num_teams, int nthread);";
  return ss.str();
}

std::string OMP4KernelHandler::DevicePointerDecl(){
  std::string DPD = "";
  llvm::raw_string_ostream ss(DPD);
  std::map<std::string,std::string> arg2data;
  if(!loop.isDirect()){
    ss << "int *map0 = arg0.map_data_d;\n"; 
    ss << "int map0size = arg0.map->dim * set_size1;\n";
  }
  for(size_t i = 0; i < loop.getNumArgs(); ++i){
    if(arg2data[loop.getArg(i).opDat] != ""){
      continue;
    } else {
      arg2data[loop.getArg(i).opDat] = "data" + std::to_string(i);
    }

    ss << loop.getArg(i).type << " *" << "data" << i << " = ";
    ss << "(" << loop.getArg(i).type << "*)" <<"arg" << i << ".data_d;\n";

    ss << "\tint " << "data" << i << "size" << " = getSetSizeFromOpArg((&arg" << i;
    ss << ") * arg" << i << ".dat->dim;\n";
  }
  return ss.str();
}

std::string OMP4KernelHandler::handleRedLocalVarDecl() {
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

std::string OMP4KernelHandler::handlelocRedToArgAssignment() {
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

std::string OMP4KernelHandler::handleOMPParLoop() {
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
