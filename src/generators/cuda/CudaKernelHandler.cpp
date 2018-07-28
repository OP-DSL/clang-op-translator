#include "CudaKernelHandler.h"
#include "core/utils.h"
#include "generators/common/handler.hpp"
#include "generators/cuda/soa/UserFuncTransformator.hpp"

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
                 hasBody(compoundStmt().bind("END")))
        .bind("cuda_func_definition");

const StatementMatcher CUDAKernelHandler::cudaFuncCallMatcher =
    callExpr(callee(functionDecl(hasName("op_cuda_skeleton"))),
             hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
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

const DeclarationMatcher CUDAKernelHandler::mapidxDeclMatcher =
    varDecl(hasName("map1idx"),
            hasAncestor(functionDecl(hasName("op_cuda_skeleton"))))
        .bind("mapidx_decl");

const StatementMatcher CUDAKernelHandler::mapidxInitMatcher =
    binaryOperator(hasLHS(declRefExpr(to(varDecl(hasName("map1idx"))))),
                   hasRHS(integerLiteral(equals(0))),
                   hasAncestor(functionDecl(hasName("op_cuda_skeleton"))))
        .bind("map1idx_init");

const StatementMatcher CUDAKernelHandler::initLocalRedArrMatcher =
    forStmt(hasBody(compoundStmt(hasAnySubstatement(binaryOperator(
                hasOperatorName("="), hasRHS(floatLiteral(equals(0.0))))))),
            hasAncestor(functionDecl(hasName("op_cuda_skeleton"))))
        .bind("init_local_arr_forstmt");

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

const StatementMatcher CUDAKernelHandler::incrementWriteMatcher =
    ifStmt(hasThen(compoundStmt(hasAnySubstatement(
               binaryOperator(hasOperatorName("+=")).bind("write_increment")))),
           hasAncestor(functionDecl(hasName("op_cuda_skeleton"))),
           hasCondition(binaryOperator(hasLHS(
               ignoringImpCasts(declRefExpr(to(varDecl(hasName("col2")))))))));

// SOA

const DeclarationMatcher CUDAKernelHandler::strideDeclMatcher =
    varDecl(hasName("direct_skeleton_stride_OP2HOST")).bind("strideDecl");

const StatementMatcher CUDAKernelHandler::strideInitMatcher =
    ifStmt(hasThen(compoundStmt(
               hasAnySubstatement(binaryOperator(hasOperatorName("="))),
               statementCountIs(1))),
           hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))),
           hasCondition(binaryOperator(
               hasLHS(ignoringImpCasts(memberExpr(member(hasName("count"))))))))
        .bind("strideInit");

const StatementMatcher CUDAKernelHandler::constantHandlingMatcher =
    compoundStmt(
        hasAnySubstatement(
            declStmt(containsDeclaration(0, varDecl(hasName("const_bytes"))))
                .bind("const_bytes_decl")),
        hasAnySubstatement(
            callExpr(callee(functionDecl(hasName("mvConstArraysToDevice"))))
                .bind("END")));

//_________________________________CONSTRUCTORS________________________________
CUDAKernelHandler::CUDAKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const clang::tooling::CompilationDatabase &Compilations,
    const OP2Application &app, size_t idx, OP2Optimizations flags)
    : Replace(Replace), Compilations(Compilations), application(app),
      loopIdx(idx), op2Flags(flags) {}

//________________________________GLOBAL_HANDLER_______________________________
void CUDAKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!lineReplHandler<FunctionDecl, 1>(Result, Replace, "user_func", [this]() {
        const ParLoop &loop = this->application.getParLoops()[loopIdx];
        std::string hostFuncText = loop.getUserFuncInc();
        if (op2Flags.SOA) {
          loop.dumpFuncTextTo("/tmp/loop.cu");
          std::string SOAresult =
              UserFuncTransformator(Compilations, loop, op2Flags).run();
          if (SOAresult != "")
            hostFuncText = SOAresult;
        }
        return "__device__ void " + loop.getName() + "_gpu" +
               hostFuncText.substr(hostFuncText.find("("));
      }))
    return; // if successfully handled return
  if (!fixEndReplHandler<FunctionDecl, CompoundStmt>(
          Result, Replace, "cuda_func_definition",
          std::bind(&CUDAKernelHandler::getCUDAFuncDefinition, this)))
    return; // if successfully handled return
  if (!HANDLER(CallExpr, 2, "func_call", CUDAKernelHandler::genFuncCall))
    return; // if successfully handled return
  if (!HANDLER(CallExpr, 2, "cuda_func_call",
               CUDAKernelHandler::genCUDAkernelLaunch))
    return; // if successfully handled return
  if (!HANDLER(VarDecl, 6, "arg0h_decl", CUDAKernelHandler::getLocalVarDecls))
    return;
  if (!HANDLER(VarDecl, 8, "mapidx_decl", CUDAKernelHandler::getMapIdxDecls))
    return;
  if (!HANDLER(BinaryOperator, 2, "map1idx_init",
               CUDAKernelHandler::getMapIdxInits))
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
  if (!HANDLER(ForStmt, 2, "init_local_arr_forstmt",
               CUDAKernelHandler::genLocalArrInit))
    return;
  if (!HANDLER(ForStmt, 1, "op_reduction_forstmt",
               CUDAKernelHandler::genRedForstmt))
    return;
  if (!HANDLER(BinaryOperator, 2, "write_increment",
               CUDAKernelHandler::genWriteIncrement))
    return;
  if (!HANDLER(VarDecl, 2, "strideDecl", CUDAKernelHandler::genStrideDecls))
    return;
  if (!HANDLER(IfStmt, 2, "strideInit", CUDAKernelHandler::genStrideInit))
    return;
  if (!fixEndReplHandler<DeclStmt, CallExpr, 0, 35>(
          Result, Replace, "const_bytes_decl", [this]() {
            const ParLoop &loop = this->application.getParLoops()[loopIdx];
            for (size_t i = 0; i < loop.getNumArgs(); ++i) {
              const OPArg &arg = loop.getArg(i);
              if (arg.isGBL && arg.accs == OP2::OP_READ)
                return "NO_REPL";
            }
            return "";
          }))
    return; // if successfully handled return
}

//___________________________________HANDLERS__________________________________

std::string CUDAKernelHandler::genStrideInit() {
  if (!op2Flags.SOA)
    return "";
  const ParLoop &loop = this->application.getParLoops()[loopIdx];
  std::string repl = "";
  llvm::raw_string_ostream os(repl);
  bool directStride = false;
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.dim > 1 &&
        ((!directStride && arg.isDirect() && !arg.isGBL) ||
         (!arg.isDirect() && loop.dat2argIdxs[loop.dataIdxs[i]] == (int)i))) {
      std::string strideName =
          (arg.isDirect() ? "direct"
                          : ("opDat" + std::to_string(loop.dataIdxs[i]))) +
          "_" + loop.getName() + "_stride_OP2";
      os << "if((OP_kernels[" << loop.getLoopID() << "].count == 1) || (";
      os << strideName << "HOST != getSetSizeFromOpArg(&arg" << i << "))) {";
      os << strideName << "HOST = getSetSizeFromOpArg(&arg" << i << ");"
         << "cudaMemcpyToSymbol(" << strideName << "CONSTANT, &" << strideName
         << "HOST, sizeof(int));";
      os << "}";
      if (arg.isDirect())
        directStride = true;
    }
  }
  return os.str();
}

std::string CUDAKernelHandler::genStrideDecls() {
  if (!op2Flags.SOA)
    return "";
  const ParLoop &loop = this->application.getParLoops()[loopIdx];
  std::string repl = "";
  llvm::raw_string_ostream os(repl);
  bool directStride = false;
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.dim > 1 &&
        ((!directStride && arg.isDirect() && !arg.isGBL) ||
         (!arg.isDirect() && loop.dat2argIdxs[loop.dataIdxs[i]] == (int)i))) {
      std::string strideName =
          (arg.isDirect() ? "direct"
                          : ("opDat" + std::to_string(loop.dataIdxs[i]))) +
          "_" + loop.getName() + "_stride_OP2";
      os << "__constant__ int " << strideName << "CONSTANT;"
         << "int " << strideName << "HOST = -1;";
      if (arg.isDirect())
        directStride = true;
    }
  }
  return os.str();
}

std::string CUDAKernelHandler::genWriteIncrement() {
  const ParLoop &loop = this->application.getParLoops()[loopIdx];
  std::string repl = "";
  llvm::raw_string_ostream os(repl);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (!arg.isDirect() && arg.accs == OP2::OP_INC) {
      int idx = loop.dataIdxs[i], mapIdx = loop.mapIdxs[i];
      std::string writeBack = "";
      for (size_t d = 0; d < arg.dim; ++d) {
        std::string argl =
            "arg" + std::to_string(i) + "_l[" + std::to_string(d) + "]";
        std::string indarg =
            "ind_arg" + std::to_string(idx) + "[" + std::to_string(d);
        if (op2Flags.SOA) {
          std::string strideName =
              (arg.isDirect() ? "direct"
                              : ("opDat" + std::to_string(loop.dataIdxs[i]))) +
              "_" + loop.getName() + "_stride_OP2CONSTANT";
          indarg += "*" + strideName;
        }
        indarg += "+map" + std::to_string(mapIdx) + "idx";
        if (!op2Flags.SOA) {
          indarg += "*" + std::to_string(arg.dim);
        }
        indarg += "]";
        os << argl << "+=" << indarg << ";";
        writeBack += indarg + "=" + argl + ";";
      }
      os << writeBack;
    }
  }

  return os.str();
}

std::string CUDAKernelHandler::getMapIdxDecls() {
  const ParLoop &loop = this->application.getParLoops()[loopIdx];
  std::string repl = "";
  llvm::raw_string_ostream os(repl);
  std::vector<int> mapinds(loop.getNumArgs(), -1);

  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isDirect() || arg.isGBL)
      continue;

    if (mapinds[loop.mapIdxs[i]] == -1) {
      mapinds[loop.mapIdxs[i]] = i;
      os << "int map" << loop.mapIdxs[i] << "idx;";
    }
  }

  return os.str();
}

std::string CUDAKernelHandler::getMapIdxInits() {
  const ParLoop &loop = this->application.getParLoops()[loopIdx];
  std::string repl = "";
  llvm::raw_string_ostream os(repl);
  std::vector<int> mapinds(loop.getNumArgs(), -1);

  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isDirect() || arg.isGBL)
      continue;

    if (mapinds[loop.mapIdxs[i]] == -1) {
      mapinds[loop.mapIdxs[i]] = i;
      os << "map" << loop.mapIdxs[i] << "idx = opDat"
         << loop.map2argIdxs[loop.arg2map[i]];
      if (op2Flags.staging == OP2::OP_COlOR2) {
        os << "Map[n + set_size *";
      } else {
        os << "Map[n + offset_b + set_size *";
      }
      os << arg.idx << "];\n";
    }
  }

  return os.str();
}

std::string CUDAKernelHandler::genCUDAkernelLaunch() {
  const ParLoop &loop = this->application.getParLoops()[loopIdx];
  std::string funcCall = "";
  std::string begin = "op_cuda_" + loop.getName() + "<<<nblocks, nthread";
  llvm::raw_string_ostream os(funcCall);
  bool hasReduction = false;
  for (const int &indirectDatArgIdx : loop.dat2argIdxs) {
    os << "(" << loop.getArg(indirectDatArgIdx).type << " *) arg"
       << indirectDatArgIdx << ".data_d,";
  }
  for (const int &mappingArgIdx : loop.map2argIdxs) {
    os << "arg" << mappingArgIdx << ".map_data_d,";
  }
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isDirect()) {
      os << "(" << arg.type << " *) args[" << i << "].data_d,";
      if (arg.isReduction())
        hasReduction = true;
    }
  }

  if (!loop.isDirect()) {
    if (op2Flags.staging == OP2::OP_COlOR2) {
      os << "start, end, Plan->col_reord,";
    } else {
      os << "block_offset,Plan->blkmap,Plan->offset,Plan->nelems,Plan->"
            "nthrcol,Plan->thrcol,Plan->ncolblk[col],set->exec_size+";
    }
  }
  os << "set->size);";
  if (hasReduction) // TODO update
    begin += ",nshared";
  return begin + ">>>(" + os.str();
}

std::string CUDAKernelHandler::genFuncCall() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string funcCall = "";
  llvm::raw_string_ostream os(funcCall);

  os << loop.getName() << "_gpu(";
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isDirect()) {
      os << "arg" << i;
      if (arg.isReduction() || (op2Flags.staging == OP2::OP_STAGE_ALL &&
                                !arg.isDirect() && arg.accs == OP2::OP_INC)) {
        os << "_l";
      } else if (!arg.isGBL) {
        if (loop.isDirect() || op2Flags.staging == OP2::OP_COlOR2) {
          os << "+n";
        } else {
          os << "+(n+offset_b)";
        }
        if (!op2Flags.SOA)
          os << "*" << arg.dim;
      }
    } else {
      if (arg.accs != OP2::OP_INC || op2Flags.staging == OP2::OP_COlOR2) {
        os << "ind_arg" << loop.dataIdxs[i] << "+map" << loop.mapIdxs[i]
           << "idx";
        if (!op2Flags.SOA)
          os << "*" << arg.dim;
      } else {
        os << "arg" << i << "_l";
      }
    }
    os << ",";
  }

  return os.str().substr(0, os.str().size() - 1) + ");";
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
    if (arg.isReduction() || (op2Flags.staging == OP2::OP_STAGE_ALL &&
                              !arg.isDirect() && arg.accs == OP2::OP_INC)) {
      std::string argstr = "arg" + std::to_string(i);
      os << arg.type << " " << argstr << "_l[" << arg.dim << "];";
    }
  }
  return os.str();
}

std::string CUDAKernelHandler::genLocalArrInit() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string localArrDecl = "";
  llvm::raw_string_ostream os(localArrDecl);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    // reduction inside function
    if (arg.isReduction() || (op2Flags.staging == OP2::OP_STAGE_ALL &&
                              !arg.isDirect() && arg.accs == OP2::OP_INC)) {
      std::string argstr = "arg" + std::to_string(i);
      std::string globidx = "[d+blockIdx.x*" + std::to_string(arg.dim) + "]";

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

  for (size_t i = 0; i < loop.dat2argIdxs.size(); ++i) {
    const OPArg &arg = loop.getArg(loop.dat2argIdxs[i]);

    if (arg.accs == OP2::OP_READ) {
      os << " const " << arg.type << " *__restrict ";
    } else {
      os << arg.type << "*";
    }
    os << "ind_arg" << i << ",";
  }
  for (const int &mappingArgIdx : loop.map2argIdxs) {
    os << "const int *__restrict opDat" << mappingArgIdx << "Map,";
  }
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    const OPArg &arg = loop.getArg(i);
    if (arg.isDirect()) {
      if (arg.accs == OP2::OP_READ) {
        os << "const " << arg.type << " *__restrict arg";
      } else {
        os << arg.type << " *arg";
      }
      os << i << ",";
    }
  }
  if (!loop.isDirect()) {
    if (op2Flags.staging == OP2::OP_COlOR2) {
      os << "int start, int end, int *col_reord,";
    } else {
      os << "int block_offset, int *blkmap, int *offset, int *nelems, int"
            "*ncolors, int *colors, int nblocks,";
    }
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
