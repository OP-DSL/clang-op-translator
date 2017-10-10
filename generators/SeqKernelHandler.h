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

  int handleUserFuncDecl(const matchers::MatchFinder::MatchResult &Result);
  int handleUserFuncCall(const matchers::MatchFinder::MatchResult &Result);
  int handleMPIReduceCall(const matchers::MatchFinder::MatchResult &Result);
  int handleMPIWaitAllIfStmt(const matchers::MatchFinder::MatchResult &Result);
  int handleMapIdxDecl(const matchers::MatchFinder::MatchResult &Result);

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
  /// @brief Matcher for op_mpi_reduce call
  static const matchers::StatementMatcher opMPIReduceMatcher;
  /// @brief Matcher for the surrounding if statement of op_mpi_wait_all calls
  static const matchers::StatementMatcher opMPIWaitAllIfStmtMatcher;
  /// @brief Matcher for the mapping declarations
  static const matchers::DeclarationMatcher mapIdxDeclMatcher;

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};

} // end of namespace OP2

#endif /* ifndef BASEKERNELHANDLER_H  */
