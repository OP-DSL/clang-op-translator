#include "VecKernelHandler.h"
#include "generators/common/handler.hpp"
#include "generators/vectorization/VecUserFuncRefTool.hpp"
#include <fstream>

namespace {
using namespace clang::ast_matchers;
const auto parLoopSkeletonCompStmtMatcher =
    compoundStmt(hasParent(functionDecl(hasName("op_par_loop_skeleton"))));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;
//___________________________________MATCHERS__________________________________
const matchers::StatementMatcher VecKernelHandler::alignedPtrMatcher =
    declStmt(containsDeclaration(0, varDecl(hasName("ptr0"), isDefinition())))
        .bind("ptr0_decl");

const StatementMatcher VecKernelHandler::vecFuncCallMatcher =
    callExpr(callee(functionDecl(hasName("skeleton_vec"), parameterCountIs(1))))
        .bind("vec_func_call");
const DeclarationMatcher VecKernelHandler::vecUserFuncMatcher =
    functionDecl(hasName("skeleton_vec"), isDefinition(), parameterCountIs(1))
        .bind("user_func_vec");
const DeclarationMatcher VecKernelHandler::localRedVarMatcher =
    varDecl(hasName("dat"),
            hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("red_dat_decl");
const DeclarationMatcher VecKernelHandler::localidx0Matcher =
    varDecl(hasName("idx0_2"),
            hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("idx0_decl");
const DeclarationMatcher VecKernelHandler::localidx1Matcher =
    varDecl(hasName("idx1_2"),
            hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("idx1_decl");
const DeclarationMatcher VecKernelHandler::localIndirectVarMatcher =
    varDecl(hasName("dat0"),
            hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("indirect_dat_decl");
const StatementMatcher VecKernelHandler::redForMatcher =
    forStmt(
        hasLoopInit(declStmt(containsDeclaration(0, varDecl(hasName("i"))))),
        hasBody(compoundStmt(statementCountIs(1),
                             hasAnySubstatement(binaryOperator()))),
        hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("red_for_stmt");
const StatementMatcher VecKernelHandler::localIndDatInitMatcher =
    binaryOperator(
        hasOperatorName("="),
        hasLHS(hasDescendant(declRefExpr(to(varDecl(hasName("dat0")))))),
        hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("localIndInit");
const StatementMatcher VecKernelHandler::localIncWriteBackMatcher =
    binaryOperator(
        hasOperatorName("+="),
        hasRHS(hasDescendant(declRefExpr(to(varDecl(hasName("dat0")))))),
        hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("local_inc_add");
const StatementMatcher VecKernelHandler::localIndRedWriteBackMatcher = forStmt(
    hasLoopInit(declStmt(containsDeclaration(0, varDecl(hasName("i"))))),
    hasBody(compoundStmt(
        statementCountIs(3),
        hasAnySubstatement(binaryOperator(hasOperatorName("="),
                                          hasLHS(hasDescendant(declRefExpr(
                                              to(varDecl(hasName("dat")))))))
                               .bind("red_write_back")))),
    hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))));

//_________________________________CONSTRUCTORS________________________________
VecKernelHandler::VecKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const clang::tooling::CompilationDatabase &Compilations,
    const OP2Application &app, size_t idx)
    : Compilations(Compilations), Replace(Replace), application(app),
      loopIdx(idx) {}

//________________________________GLOBAL_HANDLER_______________________________
void VecKernelHandler::run(const matchers::MatchFinder::MatchResult &Result) {
  if (!HANDLER(clang::FunctionDecl, 1, "user_func",
               VecKernelHandler::userFuncHandler<false>))
    return;
  if (!HANDLER(clang::DeclStmt, 2, "ptr0_decl",
               VecKernelHandler::alignedPtrDecls))
    return;
  if (!HANDLER(CallExpr, 2, "func_call", VecKernelHandler::funcCallHandler))
    return; // if successfully handled return
  if (!HANDLER(CallExpr, 2, "vec_func_call",
               VecKernelHandler::vecFuncCallHandler))
    return; // if successfully handled return
  if (!HANDLER(FunctionDecl, 2, "user_func_vec",
               VecKernelHandler::userFuncHandler<true>))
    return; // if successfully handled return
  if (!HANDLER(VarDecl, 3, "red_dat_decl",
               VecKernelHandler::localReductionVarDecls))
    return;
  if (!HANDLER(VarDecl, 2, "indirect_dat_decl",
               VecKernelHandler::localIndirectVarDecls))
    return;
  if (!HANDLER(VarDecl, 2, "idx0_decl", VecKernelHandler::handleIDX<true>))
    return;
  if (!HANDLER(VarDecl, 2, "idx1_decl", VecKernelHandler::handleIDX<false>))
    return;
  if (!HANDLER(BinaryOperator, 2, "localIndInit",
               VecKernelHandler::handleLocalIndirectInit))
    return;
  if (!HANDLER(BinaryOperator, 2, "local_inc_add",
               VecKernelHandler::handleLocalIndirectInc))
    return;
  if (!HANDLER(BinaryOperator, 3, "red_write_back",
               VecKernelHandler::handleRedWriteBack))
    return;
  if (!HANDLER(ForStmt, 2, "red_for_stmt",
               VecKernelHandler::reductionVecForStmt))
    return;
}

//___________________________________HANDLERS__________________________________

std::string VecKernelHandler::handleRedWriteBack() {
  std::string repl;
  llvm::raw_string_ostream os(repl);
  const ParLoop &loop = application.getParLoops()[loopIdx];
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isReduction()) { // TODO other types of reductions
      os << "*(" << loop.getArg(i).type << "*)arg" << loop.dat2argIdxs[i]
         << ".data += dat" << i << "[i];\n";
    }
  }
  return os.str();
}
std::string VecKernelHandler::handleLocalIndirectInc() {
  std::string repl;
  llvm::raw_string_ostream os(repl);
  const ParLoop &loop = application.getParLoops()[loopIdx];
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isDirect() ||
        loop.getArg(i).accs != OP_accs_type::OP_INC)
      continue;
    std::string linebeg = "(ptr" + std::to_string(i) + ")[idx" +
                          std::to_string(i) + "_" +
                          std::to_string(loop.getArg(i).dim) + "+";
    for (size_t d = 0; d < loop.getArg(i).dim; ++d) {
      os << linebeg << d << "]+="
         << "dat" << i << "[" << d << "][i];\n";
    }
    os << "\n";
  }
  return os.str();
}

std::string VecKernelHandler::handleLocalIndirectInit() {
  std::string repl;
  llvm::raw_string_ostream os(repl);
  const ParLoop &loop = application.getParLoops()[loopIdx];
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isDirect())
      continue;
    for (size_t d = 0; d < loop.getArg(i).dim; ++d) {
      os << "dat" << i << "[" << d << "][i] =";
      if (loop.getArg(i).accs == OP_accs_type::OP_READ) {
        os << "(ptr" << i << ")[idx" << i << "_" << loop.getArg(i).dim << "+"
           << d << "];\n";
      } else {
        os << "0;\n";
      }
    }
    os << "\n";
  }
  return os.str();
}

template <bool READS> std::string VecKernelHandler::handleIDX() {
  std::string repl;
  llvm::raw_string_ostream os(repl);
  const ParLoop &loop = application.getParLoops()[loopIdx];
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isDirect())
      continue;
    if ((READS && loop.getArg(i).accs == OP_accs_type::OP_READ) ||
        (!READS && loop.getArg(i).accs == OP_accs_type::OP_INC)) {
      int dim = loop.getArg(i).dim;
      std::string argmap =
          "arg" + std::to_string(loop.map2argIdxs[loop.arg2map[i]]) + ".map";
      os << "int idx" << i << "_" << dim << "=" << dim << "*" << argmap
         << "_data[(n+i)*" << argmap << "->dim+" << loop.getArg(i).idx << "];";
    }
  }
  return os.str();
}

std::string VecKernelHandler::localIndirectVarDecls() {
  std::string repl;
  llvm::raw_string_ostream os(repl);
  const ParLoop &loop = application.getParLoops()[loopIdx];
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (!loop.getArg(i).isDirect() && !loop.getArg(i).isGBL) {
      std::string type = loop.getArg(i).type;
      os << "ALIGNED_" << type << " " << type << " dat" << i << "["
         << loop.getArg(i).dim << "][SIMD_VEC];";
    }
  }
  return os.str();
}

template <bool VEC> std::string VecKernelHandler::userFuncHandler() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  if (VEC && loop.isDirect()) {
    return "";
  }
  std::vector<size_t> redIndexes;
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isReduction()) {
      redIndexes.push_back(i);
    }
  }
  if (!VEC && redIndexes.size() == 0) {
    return loop.getFuncText();
  }

  loop.dumpFuncTextTo("/tmp/loop.h");
  return VecUserFuncGenerator(Compilations, loop, redIndexes).run<VEC>();
}

std::string VecKernelHandler::alignedPtrDecls() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string repl = "";
  llvm::raw_string_ostream os(repl);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isGBL) {
      continue;
    }
    std::string type = loop.getArg(i).type;
    os << "ALIGNED_" << type << " ";
    if (loop.getArg(i).accs == OP_accs_type::OP_READ)
      os << "const ";
    os << type << "*__restrict__ ptr" << i << " = (" << type << " *)arg" << i
       << ".data;\n__assume_aligned(ptr" << i << ", " << type << "_ALIGN);\n";
  }
  return os.str();
}

std::string VecKernelHandler::vecFuncCallHandler() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string repl = loop.getUserFuncInfo().funcName + (loop.isDirect() ? "(" : "_vec(");
  llvm::raw_string_ostream os(repl);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (!loop.getArg(i).isGBL) {
      if (loop.getArg(i).isDirect()) {
        os << "&(ptr" << i << ")[" << loop.getArg(i).dim << "*(n+i)],";
      } else {
        os << "dat" << i << ",";
      }
    } else {
      os << "&dat" << i << "[i],";
    }
  }
  if (!loop.isDirect()) {
    os << "i,";
  }
  os.str();
  return repl.substr(0, repl.size() - 1) + ");";
}

std::string VecKernelHandler::funcCallHandler() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string repl = loop.getUserFuncInfo().funcName + "(";
  llvm::raw_string_ostream os(repl);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (!loop.getArg(i).isGBL) {
      std::string mapStr = "n";
      if (!loop.getArg(i).isDirect()) {
        mapStr = "map" + std::to_string(loop.mapIdxs[i]) + "idx";
      }
      os << "&(ptr" << i << ")[" << loop.getArg(i).dim << "*" << mapStr << "],";
    } else {
      os << loop.getArg(i).getArgCall(i, "") << ",";
    }
  }
  os.str();
  return repl.substr(0, repl.size() - 1) + ");";
}
std::string VecKernelHandler::localReductionVarDecls() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string repl;
  llvm::raw_string_ostream os(repl);
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isReduction()) {
      os << loop.getArg(i).type << " dat" << i << "[SIMD_VEC]={0.0};\n";
    }
  }
  return os.str();
}
std::string VecKernelHandler::reductionVecForStmt() {
  const ParLoop &loop = application.getParLoops()[loopIdx];
  std::string repl = "for(int i = 0; i<SIMD_VEC;++i){";
  llvm::raw_string_ostream os(repl);
  bool hasRed = false;
  for (size_t i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isReduction()) { // TODO other types of reductions
      hasRed = true;
      os << "*(" << loop.getArg(i).type << "*)arg" << i << ".data += dat" << i
         << "[i];\n";
    }
  }
  os << "}";
  return hasRed ? os.str() : "";
}

} // namespace OP2
