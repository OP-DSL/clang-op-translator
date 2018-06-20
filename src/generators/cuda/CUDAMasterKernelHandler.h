#ifndef CUDAMASTERKERNELHANDLER_HPP
#define CUDAMASTERKERNELHANDLER_HPP
#include "core/OP2WriteableRefactoringTool.hpp"
#include "core/op2_clang_core.h"
#include "generators/common/handler.hpp"
#include "clang/ASTMatchers/ASTMatchFinder.h"
namespace OP2 {

namespace matchers = clang::ast_matchers;

namespace {
const DeclarationMatcher globVarDeclMatcher =
    varDecl(hasName("global_var")).bind("global_var");
const DeclarationMatcher kernelIncludesMatcher =
    varDecl(hasName("include_placeholder")).bind("kernel_includes");
const StatementMatcher constCopyToGPUMatcher =
    ifStmt(hasCondition(
               unaryOperator(hasUnaryOperand(ignoringImpCasts(callExpr())))),
           hasAncestor(functionDecl(hasName("op_decl_const_char"))))
        .bind("if_copy");
}

class CUDAMasterKernelHandler : public matchers::MatchFinder::MatchCallback {
  const std::vector<std::string> &kernels;
  const std::set<op_global_const> &constants;

  std::map<std::string, clang::tooling::Replacements> *Replace;

public:
  CUDAMasterKernelHandler(
      std::vector<std::string> &kernels,
      const std::set<op_global_const> &constants,
      std::map<std::string, clang::tooling::Replacements> *Replace)
      : kernels(kernels), constants(constants), Replace(Replace) {}

  void addMatchersTo(clang::ast_matchers::MatchFinder &finder) {
    finder.addMatcher(globVarDeclMatcher, this);
    finder.addMatcher(kernelIncludesMatcher, this);
    finder.addMatcher(constCopyToGPUMatcher, this);
  }

  std::string generateConstDefs() {
    std::string repl;
    llvm::raw_string_ostream os(repl);
    for (const op_global_const &c : constants) {
      os << "__constant__ " << c << ";\n";
    }
    return os.str();
  }

  std::string generateConstCopies() {
    std::string repl;
    llvm::raw_string_ostream os(repl);
    for (const op_global_const &c : constants) {
      os << "if(!strcmp(name,\"" << c.name
         << "\")) { cutilSafeCall(cudaMemcpyToSymbol(" << c.name
         << ", dat, dim*size));} else ";
    }
    if (constants.size() != 0)
      os << "{printf(\"error: unknown const name\n\"); exit(1);}";
    return os.str();
  }

  std::string generateIncludes() {
    std::string repl;
    llvm::raw_string_ostream os(repl);
    for (const std::string &kernel : kernels) {
      os << "#include \"" << kernel << "\"\n";
    }
    return os.str();
  }
  virtual void run(const matchers::MatchFinder::MatchResult &Result) override {
    if (!HANDLER(clang::IfStmt, 1, "if_copy",
                 CUDAMasterKernelHandler::generateConstCopies))
      return;
    if (!HANDLER(clang::VarDecl, 20, "kernel_includes",
                 CUDAMasterKernelHandler::generateIncludes))
      return;
    if (!HANDLER(clang::VarDecl, 11, "global_var",
                 CUDAMasterKernelHandler::generateConstDefs))
      return;
  }
};

} // namespace OP2
#endif /* ifndef CUDAMASTERKERNELHANDLER_HPP */
