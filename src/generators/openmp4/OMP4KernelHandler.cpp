#include "OMP4KernelHandler.h"
#include "generators/common/ASTMatchersExtension.h"
#include "generators/common/handler.hpp"
#include "clang/AST/StmtOpenMP.h"
#include "generators/openmp4/OMP4UserFuncTransformator.hpp"

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
    varDecl(hasName("mapStart"), hasAncestor(parLoopSkeletonCompStmtMatcher))
        .bind("mapStart_decl");


//_________________________________CONSTRUCTORS________________________________
OMP4KernelHandler::OMP4KernelHandler(const clang::tooling::CompilationDatabase &Compilations,
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop, const OP2Application &application, const size_t loopIdx)
    : Compilations(Compilations), Replace(Replace), loop(loop), application(application), loopIdx(loopIdx) {}

//________________________________GLOBAL_HANDLER_______________________________
void OMP4KernelHandler::run(const MatchFinder::MatchResult &Result) {

  if (!lineReplHandler<FunctionDecl, 1>(Result, Replace, "user_func_OMP4",  [this]() {
        std::string retStr="";
        const ParLoop &loop = this->application.getParLoops()[loopIdx];
        std::string hostFuncText = loop.getUserFuncInc();
        std::vector<std::string> path = {"/tmp/omp4.cpp"};
        

        loop.dumpFuncTextTo(path[0]);

        std::string SOAresult =
            OMP4UserFuncTransformator(Compilations, loop, application, const_list, kernel_arg_name, path).run();
        if (SOAresult != "")
          hostFuncText = SOAresult;
        retStr = hostFuncText.substr(hostFuncText.find("{")) + "}\n}\n";
        return this->getmappedFunc() +  retStr;
      }))
    return; // if successfully handled return

  if (!lineReplHandler<VarDecl, 9>(Result, Replace, "mapStart_decl",  [this]() {
        return this->DevicePointerDecl();
      }))
    return; // if successfully handled return

  if (!lineReplHandler<DeclStmt, 1>(
          Result, Replace, "local_reduction_variable",
          std::bind(&OMP4KernelHandler::handleRedLocalVarDecl, this)))
    return;
  if (!HANDLER(CallExpr, 2, "func_call_OMP4", OMP4KernelHandler::handleFuncCall))
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
  std::map<std::string,std::string> arg2data;
  ss << loop.getName() << "_omp4_kernel(";
  if(!loop.isDirect()){
    ss << "map0, map0size,";
  }
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if(arg2data[loop.getArg(i).opDat] != ""){
      continue;
    } else {
      arg2data[loop.getArg(i).opDat] = "data" + std::to_string(i);
    }
    ss  << "data" << i << ", ";
    ss << "data" << i << "size, ";
  }
  if(loop.isDirect()){
    ss << "set_size, part_size != 0 ? (set->size - 1) / part_size + 1: (set->size - 1) / nthread, nthread);";
  } else {
    ss << "col_reord, set_size1, start, end, part_size != 0 ? (end - start - 1) / part_size + 1: (end - start - 1) / nthread, nthread);";
  }
  return ss.str();
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
    ss << "int " << "data" << i << "size, ";

  }
  if(!loop.isDirect()){
    arg2data.clear();
    ss << "int *col_reord, int set_size1, int start, int end, int num_teams, int nthread) {\n";
    ss << "#pragma omp target teams num_teams(num_teams) thread_limit(nthread)  \n";
    ss << "#pragma omp map(to: ";
    for (size_t i = 0; i < loop.getNumArgs(); ++i){
      if(arg2data[loop.getArg(i).opDat] != ""){
        continue;
      } else {
        arg2data[loop.getArg(i).opDat] = "data" + std::to_string(i);
      }
      ss << arg2data[loop.getArg(i).opDat] << "[0 : " << arg2data[loop.getArg(i).opDat];
      ss << "size]";
      if(i != loop.getNumArgs() -1 ){
        ss << ", ";
      }
    }
    ss << ")";
    if(const_list.size() != 0)
      ss << "\n #pragma omp map(to : ";
    for(auto it = const_list.begin() + 1; it < const_list.end(); it++){
        ss << *it;
        for (auto it_g = application.constants.begin(); it_g != application.constants.end(); it_g++){
          if(it_g->name == *it){
            ss << "[0:" << it_g->size -1<< "]"; 
          }
        }
        if(it != const_list.end() - 1){
          ss << ", ";
        }
    }
    ss << ")\n";
    ss << "#pragma omp distribute parallel for schedule(static, 1)\n";
    ss << "for ( int e=start; e<end; e++ ){\n";
    ss << "int n_op = col_reord[e];\n";
    // for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    //   ss << "int map" << i << "idx = map0[n_op + set_size1 + " << i << "\n";
    // }
    // ss << loop.getArg(i).type << " *" << "data" << i << ", ";
    // ss << "int " << "data" << i << "size, ";

  } else {
    ss << ",  int count, int num_teams, int nthread) {\n";    
    ss << "#pragma omp target teams num_teams(num_teams) thread_limit(nthread)  \n";
    ss << "#pragma omp map(";
    for (size_t i = 0; i < loop.getNumArgs(); ++i){
      ss << "to : " << arg2data[loop.getArg(i).opDat] << "[0 : " << arg2data[loop.getArg(i).opDat];
      ss << "size]";
      if(i != loop.getNumArgs() -1 ){
        ss << ", ";
      }
    }

    ss << ")";
    if(const_list.size() != 0)
      ss << "\n #pragma omp map(to : ";
    for(auto it = const_list.begin() + 1; it < const_list.end(); it++){
        ss << *it;
        for (auto it_g = application.constants.begin(); it_g != application.constants.end(); it_g++){
          if(it_g->name == *it){
            ss << "[0:" << it_g->size -1<< "]"; 
          }
        }
        if(it != const_list.end() - 1){
          ss << ", ";
        }
    }
    ss << ")\n";

    ss << "\n#pragma omp distribute parallel for schedule(static, 1)\n";
    ss << "for ( int n_op=0; n_op<count; n_op++ ){\n";
    ss << "//variable mapping\n";
    for (size_t i = 0; i < loop.getNumArgs(); ++i){
      ss << kernel_arg_name[i];
      ss << " = &" << arg2data[loop.getArg(i).opDat] << "[" << loop.getArg(i).idx << " * n_op];\n";
    }
  }
  return ss.str();
}

std::string OMP4KernelHandler::DevicePointerDecl(){
  std::string DPD = "";
  llvm::raw_string_ostream ss(DPD);
  std::map<std::string,std::string> arg2data;
  if(!loop.isDirect()){
    ss << "int *map0 = arg0.map_data_d;\n"; 
    ss << "int map0size = arg0.map->dim * set_size1;\n\n";
  }
  for(size_t i = 0; i < loop.getNumArgs(); ++i){
    if(arg2data[loop.getArg(i).opDat] != ""){
      continue;
    } else {
      arg2data[loop.getArg(i).opDat] = "data" + std::to_string(i);
    }

    ss << loop.getArg(i).type << " *" << "data" << i << " = ";
    ss << "(" << loop.getArg(i).type << "*)" <<"arg" << i << ".data_d;\n";

    ss << "int " << "data" << i << "size" << " = getSetSizeFromOpArg(&arg" << i;
    ss << ") * arg" << i << ".dat->dim;\n\n";
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
