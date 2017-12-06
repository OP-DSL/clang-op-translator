#ifndef KERNELHANDLERSKELETON_HPP
#define KERNELHANDLERSKELETON_HPP
#include "core/OPParLoopData.h"
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

  std::string handleRedWriteBack();

  std::string handleLocalIndirectInc();

  std::string handleLocalIndirectInit();

  template <bool READS> std::string handleIDX();

  std::string localIndirectVarDecls();

  std::string funcDeclCopy();

  std::string userFuncVecHandler();

  std::string alignedPtrDecls();

  std::string vecFuncCallHandler();

  std::string funcCallHandler();

  std::string localReductionVarDecls();

  std::string reductionVecForStmt();

public:
  VecKernelHandler(std::map<std::string, clang::tooling::Replacements> *Replace,
                   const clang::tooling::CompilationDatabase &Compilations,
                   const ParLoop &loop);

  // Static matchers handled by this class
  /// @brief Matcher for the declaration of ptr0 in the skeleton
  static const matchers::StatementMatcher alignedPtrMatcher;
  static const matchers::StatementMatcher vecFuncCallMatcher;
  static const matchers::StatementMatcher redForMatcher;
  static const matchers::DeclarationMatcher vecUserFuncMatcher;
  static const matchers::DeclarationMatcher localRedVarMatcher;
  static const matchers::DeclarationMatcher localIndirectVarMatcher;
  static const matchers::DeclarationMatcher localidx0Matcher;
  static const matchers::DeclarationMatcher localidx1Matcher;
  static const matchers::StatementMatcher localIndDatInitMatcher;
  static const matchers::StatementMatcher localIncWriteBackMatcher;
  static const matchers::StatementMatcher localIndRedWriteBackMatcher;

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};
} // namespace OP2
#endif /* ifndef KERNELHANDLERSKELETON_HPP */
