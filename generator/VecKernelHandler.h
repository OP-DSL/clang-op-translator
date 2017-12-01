#ifndef KERNELHANDLERSKELETON_HPP
#define KERNELHANDLERSKELETON_HPP
#include "../OPParLoopData.h"
#include "VecDirectUserFuncRefTool.hpp"
#include "handler.hpp"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

class VecKernelHandler
    : public clang::ast_matchers::MatchFinder::MatchCallback {
protected:
  const clang::tooling::CompilationDatabase &Compilations;
  std::map<std::string, clang::tooling::Replacements> *Replace;
  const ParLoop &loop;

  std::string funcDeclCopy() {
    std::vector<size_t> redIndexes;
    for (size_t i = 0; i < loop.getNumArgs(); ++i) {
      if (loop.getArg(i).isReduction()) {
        redIndexes.push_back(i);
      }
    }

    if (redIndexes.size() == 0) {
      return loop.getFuncText();
    }
    return VecDirectUserFuncGenerator(Compilations, loop, redIndexes).run();
  }

  std::string userFuncVecHandler() {
    if (loop.isDirect()) {
      return "";
    }
    return "";
  }
  std::string alignedPtrDecls() {
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

  std::string vecFuncCallHandler() {
    std::string repl = loop.getName() + (loop.isDirect() ? "(" : "_vec(");
    llvm::raw_string_ostream os(repl);
    for (size_t i = 0; i < loop.getNumArgs(); ++i) {
      if (!loop.getArg(i).isGBL) {
        os << "&(ptr" << i << ")[" << loop.getArg(i).dim << "*(n+i)],";
      } else {
        os << "&dat" << i << "[i],";
      }
    }
    os.str();
    return repl.substr(0, repl.size() - 1) + ");";
  }

  std::string funcCallHandler() {
    std::string repl = loop.getName() + "(";
    llvm::raw_string_ostream os(repl);
    for (size_t i = 0; i < loop.getNumArgs(); ++i) {
      if (!loop.getArg(i).isGBL) {
        os << "&(ptr" << i << ")[" << loop.getArg(i).dim << "*n],";
      } else {
        os << loop.getArg(i).getArgCall(i, "") << ",";
      }
    }
    os.str();
    return repl.substr(0, repl.size() - 1) + ");";
  }
  std::string localReductionVarDecls() {
    std::string repl;
    llvm::raw_string_ostream os(repl);
    for (size_t i = 0; i < loop.getNumArgs(); ++i) {
      if (loop.getArg(i).isReduction()) {
        os << loop.getArg(i).type << " dat" << i << "[SIMD_VEC]={0.0};\n";
      }
    }
    return os.str();
  }
  std::string reductionVecForStmt() {
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

public:
  VecKernelHandler(std::map<std::string, clang::tooling::Replacements> *Replace,
                   const clang::tooling::CompilationDatabase &Compilations,
                   const ParLoop &loop)
      : Compilations(Compilations), Replace(Replace), loop(loop) {}

  // Static matchers handled by this class
  /// @brief Matcher for the declaration of ptr0 in the skeleton
  static const matchers::StatementMatcher alignedPtrMatcher;
  static const matchers::StatementMatcher vecFuncCallMatcher;
  static const matchers::StatementMatcher redForMatcher;
  static const matchers::DeclarationMatcher vecUserFuncMatcher;
  static const matchers::DeclarationMatcher localRedVarMatcher;

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override {
    if (!HANDLER(clang::FunctionDecl, 1, "user_func",
                 VecKernelHandler::funcDeclCopy))
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
                 VecKernelHandler::userFuncVecHandler))
      return; // if successfully handled return
    if (!HANDLER(VarDecl, 3, "red_dat_decl",
                 VecKernelHandler::localReductionVarDecls))
      return;
    if (!HANDLER(ForStmt, 2, "red_for_stmt",
                 VecKernelHandler::reductionVecForStmt))
      return;
  }
};

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
const StatementMatcher VecKernelHandler::redForMatcher =
    forStmt(
        hasLoopInit(declStmt(containsDeclaration(0, varDecl(hasName("i"))))),
        hasBody(compoundStmt(statementCountIs(1),
                             hasAnySubstatement(binaryOperator()))),
        hasAncestor(functionDecl(hasName("op_par_loop_skeleton"))))
        .bind("red_for_stmt");
} // namespace OP2
#endif /* ifndef KERNELHANDLERSKELETON_HPP */
