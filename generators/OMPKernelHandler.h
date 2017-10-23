#ifndef SEQEKERNELHANDLER_H
#define SEQEKERNELHANDLER_H
#include "../OPParLoopData.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

/// @brief Callback for perform the specific modifications for OpenMP
/// kernels on op_par_loop_skeleton
///
/// Callback for perform the modifications for OpenMP kernels on
/// op_par_loop_skeleton e.g. replace dummy skeleton function to user function,
/// chang the function call for user function
class OMPKernelHandler : public matchers::MatchFinder::MatchCallback {
protected:
  std::map<std::string, clang::tooling::Replacements> *Replace;
  const ParLoop &loop;

  int handleRedLocalVarDecl(const matchers::MatchFinder::MatchResult &Result);

public:
  /// @brief Construct a OMPKernelHandler
  ///
  /// @param Replace Replacements map from the RefactoringTool where
  /// Replacements should added.
  ///
  /// @param loop The ParLoop that the file is currently generated.
  OMPKernelHandler(std::map<std::string, clang::tooling::Replacements> *Replace,
                   const ParLoop &loop);

  // Static matchers handled by this class
  /// @brief Matcher for the placeholder of the user funciton
  static const matchers::StatementMatcher locRedVarMatcher;

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};

} // end of namespace OP2

#endif /* ifndef BASEKERNELHANDLER_H  */
