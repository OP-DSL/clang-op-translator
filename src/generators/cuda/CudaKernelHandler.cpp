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

const DeclarationMatcher CUDAKernelHandler::hostReductArrsMatcher =
    varDecl(hasName("reduct_bytes"),
            hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("reduct_bytes");

const StatementMatcher CUDAKernelHandler::mvReductCallMatcher =
    callExpr(callee(functionDecl(hasName("mvReductArraysToHost"),
                                 parameterCountIs(1))),
             hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("mvReductArraysToHost");
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
  if (!HANDLER(FunctionDecl, 1, "cuda_func_definition",
               CUDAKernelHandler::getCUDAFuncDefinition))
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
              os << "(" << arg.type << " *) arg" << i << ".data_d,";
              if (arg.isReduction())
                hasReduction = true;
            }
            os << "set->size);";
            if (hasReduction)
              begin += ",nshared";
            return begin + ">>>(" + os.str();
          }))
    return; // if successfully handled return
  if (!HANDLER(VarDecl, 6, "arg0h_decl",
               CUDAKernelHandler::getLocalVarDecls4Reduct))
    return;
  if (!HANDLER(VarDecl, 2, "reduct_bytes",
               CUDAKernelHandler::getReductArrsToDevice))
    return;
  if (!HANDLER(CallExpr, 2, "mvReductArraysToHost",
               CUDAKernelHandler::getHostReduction))
    return;
}

//___________________________________HANDLERS__________________________________

std::string CUDAKernelHandler::getCUDAFuncDefinition() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string funcDef = "__global__ void op_cuda_" + loop.getName() + "(";
  llvm::raw_string_ostream os(funcDef);

  // local helper vars for reduction handling
  std::string localVars4Red = "", globalRed = "";
  llvm::raw_string_ostream localOS(localVars4Red), globalRedOS(globalRed);

  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    // function parameters
    if (arg.accs == OP2::OP_READ) {
      os << "const " << arg.type << " *__restrict arg";
    } else {
      os << arg.type << " *arg";
    }
    os << i << ",";

    // reduction inside function

    if (arg.isReduction()) {
      std::string argstr = "arg" + std::to_string(i);
      std::string globidx = "[d+blockIdx.x*" + std::to_string(arg.dim) + "]";

      localOS << arg.type << " " << argstr << "_l[" << arg.dim << "];";
      localOS << "for(int d=0;d<" << arg.dim << ";++d){";
      localOS << argstr << "_l[d] = ";
      if (arg.accs == OP2::OP_INC) {
        localOS << "ZERO_" << arg.type << ";";
      } else { // min/max red
        localOS << argstr << globidx << ";";
      }
      localOS << "}";

      localOS.str();

      globalRedOS << "for(int d=0;d<" << arg.dim << ";++d){";
      globalRedOS << "op_reduction<" << arg.accs << ">(&" << argstr << globidx
                  << ", " << argstr << "_l[d]);";
      globalRedOS << "}";
      globalRedOS.str();
    }
  }

  os << "int set_size) {"
     << "int n = threadIdx.x + blockIdx.x * blockDim.x;";

  os << localVars4Red;

  os << "if (n < set_size) {" << loop.getName() << "_gpu(";
  if (loop.getArg(0).isReduction()) {
    os << "arg0_l";
  } else {
    os << "arg0+n*" << loop.getArg(0).dim;
  }
  for (size_t i = 1; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    os << ",arg" << i;
    if (arg.isReduction()) {
      os << "_l";
    } else {
      os << "+n*" << arg.dim;
    }
  }

  os << ");}"; // close if(n<set_size)
  os << globalRed;
  return os.str() + "}"; // close function def
}

std::string CUDAKernelHandler::getLocalVarDecls4Reduct() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string varDecls = "";
  llvm::raw_string_ostream os(varDecls);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isReduction()) {
      std::string argi = "arg" + std::to_string(i);
      os << arg.type << "* " << argi << "h = (" << arg.type << " *)" << argi
         << ".data;";
    }
  }

  return os.str();
}

std::string CUDAKernelHandler::getReductArrsToDevice() {
  const ParLoop &loop = application.getParLoops()[loopIdx];

  const std::string begin =
      "int maxblocks = nblocks;int reduct_bytes = 0;int reduct_size  = 0;";

  std::string redString = "", initRedArrs = "";
  llvm::raw_string_ostream redBytesOS(redString), initArrOs(initRedArrs);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isReduction()) {
      redBytesOS << "reduct_bytes += ROUND_UP(maxblocks*" << arg.dim
                 << "*sizeof(" << arg.type << "));"
                 << "reduct_size   = MAX(reduct_size,sizeof(" << arg.type
                 << "));";
      std::string argi = "arg" + std::to_string(i);
      initArrOs << argi << ".data   = OP_reduct_h + reduct_bytes;";
      initArrOs << argi << ".data_d = OP_reduct_d + reduct_bytes;";
      initArrOs << "for( int b=0; b<maxblocks; b++ ){"
                << "for( int d=0; d<" << arg.dim << "; ++d){ ((" << arg.type
                << "*)" << argi << ".data)[d+b*" << arg.dim << "]=";
      if (arg.accs == OP2::OP_INC) {
        initArrOs << "ZERO_" << arg.type << ";";
      } else { // min/max red
        initArrOs << argi << "h[d];";
      }
      initArrOs << "}}";
      initArrOs << "reduct_bytes += ROUND_UP(maxblocks*" << arg.dim
                << "*sizeof(" << arg.type << "));";
    }
  }
  if (redBytesOS.str() == "")
    return "";

  return begin + redString +
         "reallocReductArrays(reduct_bytes);reduct_bytes=0;" + initArrOs.str() +
         "mvReductArraysToDevice(reduct_bytes);int nshared = "
         "reduct_size*nthread;";
}

std::string CUDAKernelHandler::getHostReduction() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string begin = "//transfer global reduction data back to "
                      "CPU\nmvReductArraysToHost(reduct_bytes);";
  std::string copyCalls = "";
  llvm::raw_string_ostream os(copyCalls);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isReduction()) {
      const OPArg &arg = loop.getArg(i);
      os << "for(int b=0;b<maxblocks;b++){";
      os << "for(int d=0;d<" << arg.dim << ";d++){";
      std::string argi = "arg" + std::to_string(i);
      os << argi << "h[d] = ";
      if (arg.accs == OP2::OP_INC) {
        os << argi << "h[d] + ((" << arg.type << " *)" << argi << ".data)[d+b*"
           << arg.dim << "];";
      } else {
        os << ((arg.accs == OP2::OP_MIN) ? "MIN(" : "MAX(") << argi
           << "h[d], ((" << arg.type << " *)" << argi << ".data)[d+b*"
           << arg.dim << "]);";
      }
      os << "}}" << argi << ".data= (char *)" << argi << "h;";
      os << "op_mpi_reduce(&" << argi << "," << argi << "h);";
    }
  }

  if (os.str() == "")
    return "";

  return begin + copyCalls;
}

} // namespace OP2
