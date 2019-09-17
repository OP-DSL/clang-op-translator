#ifndef OMP4KernelHandler_H
#define OMP4KernelHandler_H
#include "core/OPParLoopData.h"
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
class OMP4KernelHandler : public matchers::MatchFinder::MatchCallback {
protected:
  std::map<std::string, clang::tooling::Replacements> *Replace;
  const ParLoop &loop;

  std::string handleRedLocalVarDecl();
  std::string handlelocRedToArgAssignment();
  std::string handleFuncCall();
  std::string getmappedFunc();
  std::string handleOMPParLoop();
  std::string DevicePointerDecl();
  std::string AssignbackReduction();

  const OP2Application &application;
  const clang::tooling::CompilationDatabase &Compilations;
  std::vector <std::string> const_list;
  std::vector <std::string> kernel_arg_name;
  const size_t loopIdx;

public:
  /// @brief Construct a OMP4KernelHandler
  ///
  /// @param Replace Replacements map from the RefactoringTool where
  /// Replacements should added.
  ///
  /// @param loop The ParLoop that the file is currently generated.
  OMP4KernelHandler(const clang::tooling::CompilationDatabase &Compilations, std::map<std::string, clang::tooling::Replacements> *Replace,
                   const ParLoop &loop, const OP2Application &application, const size_t loopIdx);

  // Static matchers handled by this class
  /// @brief Matcher for the declaration of local variables for OpenMP reduction
  static const matchers::StatementMatcher locRedVarMatcher;
  /// @brief Matcher that matches the assignment of  local reduction result to
  /// op_arg
  static const matchers::StatementMatcher locRedToArgMatcher;
  /// @brief Matcher that matches the omp parallel for pragma
  static const matchers::StatementMatcher ompParForMatcher;

  /// @brief Matcher for the placeholder of the user function
  static const matchers::DeclarationMatcher userFuncMatcher;
  /// @brief Matcher for function kall in kernel
  static const matchers::StatementMatcher funcCallMatcher;
  /// @brief Matcher for the mapping declarations
  static const matchers::DeclarationMatcher mapIdxDeclMatcher;


  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};

} // end of namespace OP2

#endif /* ifndef OMP4KernelHandler_H  */
