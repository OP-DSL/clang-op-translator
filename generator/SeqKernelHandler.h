#ifndef SEQEKERNELHANDLER_H
#define SEQEKERNELHANDLER_H
#include "../OPParLoopData.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

/// @brief Callback for perform the specific modifications for sequential
/// kernels on op_par_loop_skeleton
///
/// Callback for perform the modifications for sequential kernels on
/// op_par_loop_skeleton e.g. replace dummy skeleton function to user function,
/// chang the function call for user function
class SeqKernelHandler : public matchers::MatchFinder::MatchCallback {
protected:
  std::map<std::string, clang::tooling::Replacements> *Replace;
  const ParLoop &loop;

public:
  /// @brief Construct a SeqKernelHandler
  ///
  /// @param Replace Replacements map from the RefactoringTool where
  /// Replacements should added.
  ///
  /// @param loop The ParLoop that the file is currently generated.
  SeqKernelHandler(std::map<std::string, clang::tooling::Replacements> *Replace,
                   const ParLoop &loop);
  // Static matchers handled by this class
  /// @brief Matcher for the placeholder of the user funciton
  static const matchers::DeclarationMatcher userFuncMatcher;
  /// @brief Matcher for function kall in kernel
  static const matchers::StatementMatcher funcCallMatcher;
  /// @brief Matcher for the mapping declarations
  static const matchers::DeclarationMatcher mapIdxDeclMatcher;

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};

} // end of namespace OP2

#endif /* ifndef SEQKERNELHANDLER_H  */
