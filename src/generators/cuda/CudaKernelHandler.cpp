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
            std::string funcCall =
                "op_cuda_" + loop.getName() + "<<<nblocks, nthread>>>(";
            llvm::raw_string_ostream os(funcCall);
            for (size_t i = 0; i < loop.getNumArgs(); ++i) {
              const OPArg &arg = loop.getArg(i);
              os << "(" << arg.type << " *) arg" << i << ".data_d,";
            }
            os << "set->size);";
            return os.str();
          }))
    return; // if successfully handled return
}

//___________________________________HANDLERS__________________________________

std::string CUDAKernelHandler::getCUDAFuncDefinition() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string funcDef = "__global__ void op_cuda_" + loop.getName() + "(";
  llvm::raw_string_ostream os(funcDef);

  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.accs == OP2::OP_READ) {
      os << "const " << arg.type << " *__restrict arg";
    } else {
      os << arg.type << " *arg";
    }
    os << i << ",";
  }

  os << "int set_size) {"
     << "int n = threadIdx.x + blockIdx.x * blockDim.x;"
     << "if (n < set_size) {" << loop.getName() << "_gpu("
     << "arg0+n*" << loop.getArg(0).dim;

  for (size_t i = 1; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    os << ",arg" << i << "+n*" << arg.dim;
  }

  os << ");}";           // close if(n<set_size)
  return os.str() + "}"; // close function def
}

} // namespace OP2
