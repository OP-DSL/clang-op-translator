#ifndef CUDAKERNELHANDLER_H
#define CUDAKERNELHANDLER_H
#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

/// @brief Callback for perform the specific modifications for CUDA
/// kernels on skeleton
class CUDAKernelHandler : public matchers::MatchFinder::MatchCallback {
protected:
  std::map<std::string, clang::tooling::Replacements> *Replace;
  const OP2Application &application;
  const size_t loopIdx;
  Staging staging;

  std::string getCUDAFuncDefinition();
  std::string getLocalVarDecls();
  template <bool GEN_CONSTANTS = false> std::string getReductArrsToDevice();
  std::string getHostReduction();
  std::string genLocalArrDecl();
  std::string genLocalArrInit();
  std::string genRedForstmt();
  std::string genFuncCall();
  std::string genCUDAkernelLaunch();
  std::string getMapIdxDecls();

public:
  /// @brief Construct a CUDAKernelHandler
  ///
  /// @param Replace Replacements map from the RefactoringTool where
  /// Replacements should added.
  /// @param app Application collected data
  /// @param idx index of the currently generated loop
  CUDAKernelHandler(
      std::map<std::string, clang::tooling::Replacements> *Replace,
      const OP2Application &app, size_t idx, Staging staging);
  // Static matchers handled by this class
  static const matchers::DeclarationMatcher cudaFuncMatcher;
  static const matchers::StatementMatcher cudaFuncCallMatcher;
  static const matchers::DeclarationMatcher declLocalRedArrMatcher;
  static const matchers::StatementMatcher initLocalRedArrMatcher;
  static const matchers::StatementMatcher opReductionMatcher;
  static const matchers::StatementMatcher updateRedArrsOnHostMatcher;
  static const matchers::StatementMatcher setReductionArraysToArgsMatcher;
  static const matchers::StatementMatcher setConstantArraysToArgsMatcher;
  static const matchers::DeclarationMatcher arg0hDeclMatcher;
  static const matchers::DeclarationMatcher mapidxDeclMatcher;

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};

} // end of namespace OP2

#endif /* ifndef CUDAKERNELHANDLER_H  */
