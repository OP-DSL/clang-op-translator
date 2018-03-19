#ifndef CUDAKERNELHANDLER_H
#define CUDAKERNELHANDLER_H
#include "core/OPParLoopData.h"
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

public:
  /// @brief Construct a CUDAKernelHandler
  ///
  /// @param Replace Replacements map from the RefactoringTool where
  /// Replacements should added.
  /// @param app Application collected data
  /// @param idx index of the currently generated loop
  CUDAKernelHandler(
      std::map<std::string, clang::tooling::Replacements> *Replace,
      const OP2Application &app, size_t idx);
  // Static matchers handled by this class

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};

} // end of namespace OP2

#endif /* ifndef CUDAKERNELHANDLER_H  */
