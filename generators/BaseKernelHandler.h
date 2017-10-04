#ifndef BASEKERNELHANDLER_H
#define BASEKERNELHANDLER_H
#include "../op_par_loop.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

/// @brief Callback for perform the base modifications on op_par_loop_skeleton
///
/// Callback for perform the base modifications on op_par_loop_skeleton e.g.
/// change name, add user function decl, set args[], and modifications came
/// from numper of args.
class BaseKernelHandler : public matchers::MatchFinder::MatchCallback {
protected:
  std::map<std::string, clang::tooling::Replacements> *Replace;
  const ParLoop &loop;
  int handleParLoopDecl(const matchers::MatchFinder::MatchResult &Result);
  int handleNargsDecl(const matchers::MatchFinder::MatchResult &Result);
  int handleArgsArrDecl(const matchers::MatchFinder::MatchResult &Result);
  int handleArgsArrSetter(const matchers::MatchFinder::MatchResult &Result);
  int handleOPTimingRealloc(const matchers::MatchFinder::MatchResult &Result);

public:
  /// @brief Construct a BaseKernelHandler
  ///
  /// @param Replace Replacements map from the RefactoringTool where
  /// Replacements should added.
  ///
  /// @param loop The ParLoop that the file is currentlz generated.
  BaseKernelHandler(
      std::map<std::string, clang::tooling::Replacements> *Replace,
      const ParLoop &loop);
  // Static matchers handled by this class
  /// @brief Matcher for the op_par_loop_skeleton declaration
  static const matchers::DeclarationMatcher parLoopDeclMatcher;
  /// @brief Matcher for the declaration of nargs
  static const matchers::DeclarationMatcher nargsMatcher;
  /// @brief Matcher for the declaration of args array
  static const matchers::DeclarationMatcher argsArrMatcher;
  /// @brief Matcher for filling args array with op_args
  static const matchers::StatementMatcher argsArrSetterMatcher;
  /// @brief Matcher for op_timing_realloc call to change kernel id
  static const matchers::StatementMatcher opTimingReallocMatcher;

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};

} // end of namespace OP2

#endif /* ifndef BASEKERNELHANDLER_H  */
