#include "CudaKernelHandler.h"
#include "core/utils.h"
#include "generators/common/handler.hpp"

namespace {
using namespace clang::ast_matchers;
const auto parLoopSkeletonCompStmtMatcher =
    compoundStmt(hasParent(functionDecl(hasName("op_par_loop_skeleton"))));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;

//___________________________________MATCHERS__________________________________
const DeclarationMatcher CUDAKernelHandler::cudaFuncMatcher =
    functionDecl(hasName("op_cuda_skeleton"), isDefinition(),
                 parameterCountIs(2))
        .bind("cuda_func_definition");

const StatementMatcher CUDAKernelHandler::cudaFuncCallMatcher =
    callExpr(
        callee(functionDecl(hasName("op_cuda_skeleton"), parameterCountIs(2))))
        .bind("cuda_func_call");

const DeclarationMatcher CUDAKernelHandler::arg0hDeclMatcher =
    varDecl(hasName("arg0h"),
            hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("arg0h_decl");

const StatementMatcher CUDAKernelHandler::setReductionArraysToArgsMatcher =
    callExpr(callee(functionDecl(hasName("setRedArrToArg"))),
             hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("initRedArrs");

const DeclarationMatcher CUDAKernelHandler::declLocalRedArrMatcher =
    varDecl(hasName("arg0_l"),
            hasAncestor(functionDecl(hasName("op_cuda_skeleton"))))
        .bind("arg0_l_decl");

const StatementMatcher CUDAKernelHandler::opReductionMatcher =
    forStmt(hasBody(compoundStmt(hasAnySubstatement(
                callExpr(callee(functionDecl(hasName("op_reduction"))))))))
        .bind("op_reduction_forstmt");

const StatementMatcher CUDAKernelHandler::setConstantArraysToArgsMatcher =
    callExpr(callee(functionDecl(hasName("setConstantArrToArg"))),
             hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("initConstArrs");

const StatementMatcher CUDAKernelHandler::updateRedArrsOnHostMatcher =
    callExpr(
        callee(functionDecl(hasName("updateRedArrToArg"), parameterCountIs(3))),
        hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("updateRedHost");
//_________________________________CONSTRUCTORS________________________________
CUDAKernelHandler::CUDAKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const OP2Application &app, size_t idx)
    : Replace(Replace), application(app), loopIdx(idx) {}

//________________________________GLOBAL_HANDLER_______________________________
void CUDAKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!lineReplHandler<FunctionDecl, 1>(Result, Replace, "user_func", [this]() {
        const ParLoop &loop = this->application.getParLoops()[loopIdx];
        std::string hostFuncText = loop.getUserFuncInc();
        return "__device__ void " + loop.getName() + "_gpu" +
               hostFuncText.substr(hostFuncText.find("("));
      }))
    return; // if successfully handled return
  if (!fixLengthReplHandler<FunctionDecl>(
          Result, Replace, "cuda_func_definition", 61,
          std::bind(&CUDAKernelHandler::getCUDAFuncDefinition, this)))
    return; // if successfully handled return
  if (!HANDLER(CallExpr, 2, "func_call", CUDAKernelHandler::genFuncCall))
    return; // if successfully handled return
  if (!lineReplHandler<CallExpr, 2>(
          Result, Replace, "cuda_func_call", [this]() {
            const ParLoop &loop = this->application.getParLoops()[loopIdx];
            std::string funcCall = "";
            std::string begin =
                "op_cuda_" + loop.getName() + "<<<nblocks, nthread";
            llvm::raw_string_ostream os(funcCall);
            bool hasReduction = false;
            for (size_t i = 0; i < loop.getNumArgs(); ++i) {
              const OPArg &arg = loop.getArg(i);
              os << "(" << arg.type << " *) args[" << i << "].data_d,";
              if (arg.isReduction())
                hasReduction = true;
            }
            os << "set->size);";
            if (hasReduction)
              begin += ",nshared";
            return begin + ">>>(" + os.str();
          }))
    return; // if successfully handled return
  if (!HANDLER(VarDecl, 6, "arg0h_decl", CUDAKernelHandler::getLocalVarDecls))
    return;
  if (!HANDLER(CallExpr, 2, "initRedArrs",
               CUDAKernelHandler::getReductArrsToDevice<false>))
    return;
  if (!HANDLER(CallExpr, 2, "initConstArrs",
               CUDAKernelHandler::getReductArrsToDevice<true>))
    return;
  if (!HANDLER(CallExpr, 2, "updateRedHost",
               CUDAKernelHandler::getHostReduction))
    return;
  if (!HANDLER(VarDecl, 2, "arg0_l_decl", CUDAKernelHandler::genLocalArrDecl))
    return;
  if (!HANDLER(ForStmt, 1, "op_reduction_forstmt",
               CUDAKernelHandler::genRedForstmt))
    return;
}

//___________________________________HANDLERS__________________________________

std::string CUDAKernelHandler::genFuncCall() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string funcCall = "";
  llvm::raw_string_ostream os(funcCall);

  os << loop.getName() << "_gpu(arg0";
  if (loop.getArg(0).isReduction()) {
    os << "_l";
  } else if (!loop.getArg(0).isGBL) {
    os << "+n*" << loop.getArg(0).dim;
  }
  for (size_t i = 1; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    os << ",arg" << i;
    if (arg.isReduction()) {
      os << "_l";
    } else if (!arg.isGBL) {
      os << "+n*" << arg.dim;
    }
  }

  os << ");";

  return os.str();
}

std::string CUDAKernelHandler::genRedForstmt() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string repl = "";
  llvm::raw_string_ostream os(repl);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isReduction()) {
      std::string argstr = "arg" + std::to_string(i);
      std::string globidx = "[d+blockIdx.x*" + std::to_string(arg.dim) + "]";

      os << "for(int d=0;d<" << arg.dim << ";++d){";
      os << "op_reduction<" << arg.accs << ">(&" << argstr << globidx << ", "
         << argstr << "_l[d]);";
      os << "}";
    }
  }
  return os.str();
}

std::string CUDAKernelHandler::genLocalArrDecl() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string localArrDecl = "";
  llvm::raw_string_ostream os(localArrDecl);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    // reduction inside function
    if (arg.isReduction()) {
      std::string argstr = "arg" + std::to_string(i);
      std::string globidx = "[d+blockIdx.x*" + std::to_string(arg.dim) + "]";

      os << arg.type << " " << argstr << "_l[" << arg.dim << "];";
      os << "for(int d=0;d<" << arg.dim << ";++d){" << argstr << "_l[d] = ";
      if (arg.accs == OP2::OP_INC) {
        os << "ZERO_" << arg.type << ";";
      } else { // min/max red
        os << argstr << globidx << ";";
      }
      os << "}";
    }
  }
  return os.str();
}

std::string CUDAKernelHandler::getCUDAFuncDefinition() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string funcDef = "__global__ void op_cuda_" + loop.getName() + "(";
  llvm::raw_string_ostream os(funcDef);

  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    // function parameters
    if (arg.accs == OP2::OP_READ) {
      os << "const " << arg.type << " *__restrict arg";
    } else {
      os << arg.type << " *arg";
    }
    os << i << ",";
  }

  os << "int set_size)";
  return os.str();
}

std::string CUDAKernelHandler::getLocalVarDecls() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string varDecls = "";
  llvm::raw_string_ostream os(varDecls);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isGBL) {
      std::string argi = "arg" + std::to_string(i);
      os << arg.type << "* " << argi << "h = (" << arg.type << " *)" << argi
         << ".data;";
    }
  }

  return os.str();
}

template <bool GEN_CONSTANTS>
std::string CUDAKernelHandler::getReductArrsToDevice() {
  const ParLoop &loop = application.getParLoops()[loopIdx];

  std::string initRedArrs = "";
  llvm::raw_string_ostream initArrOs(initRedArrs);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (!GEN_CONSTANTS && arg.isReduction()) {
      std::string argi = "arg" + std::to_string(i);
      if (arg.accs == OP2::OP_INC) {
        initArrOs << arg.type << " " << argi << "_ZERO = ZERO_" << arg.type
                  << ";";
      }
      initArrOs << "setRedArrToArg<" << arg.type << "," << arg.accs << ">(args["
                << i << "],maxblocks,";
      if (arg.accs == OP_INC) {
        initArrOs << "&" << argi << "_ZERO);";
      } else {
        initArrOs << argi << "h);";
      }
    } else if (GEN_CONSTANTS && arg.isGBL && arg.accs == OP2::OP_READ) {
      initArrOs << "setConstantArrToArg<" << arg.type << ">(args[" << i
                << "],arg" << i << "h);";
    }
  }
  return initArrOs.str();
}

std::string CUDAKernelHandler::getHostReduction() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string begin = "//Perform global reduction on CPU\n";
  std::string copyCalls = "";
  llvm::raw_string_ostream os(copyCalls);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isReduction()) {
      const OPArg &arg = loop.getArg(i);
      std::string argi = "arg" + std::to_string(i);
      os << "updateRedArrToArg<" << arg.type << "," << arg.accs << ">(args["
         << i << "],maxblocks," << argi << "h);";
    }
  }

  if (os.str() == "")
    return "";

  return begin + copyCalls;
}

} // namespace OP2
