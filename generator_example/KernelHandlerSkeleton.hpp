#ifndef KERNELHANDLERSKELETON_HPP
#define KERNELHANDLERSKELETON_HPP
#include "../OPParLoopData.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

class KernelHandlerSkeleton
    : public clang::ast_matchers::MatchFinder::MatchCallback {
protected:
  std::map<std::string, clang::tooling::Replacements> *Replace;
  const ParLoop &loop;

public:
  KernelHandlerSkeleton(
      std::map<std::string, clang::tooling::Replacements> *Replace,
      const ParLoop &loop);

  // Static matchers handled by this class
  /// @brief static matchers
  static const matchers::DeclarationMatcher ExampleMatcher;

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace OP2
#endif /* ifndef KERNELHANDLERSKELETON_HPP */
