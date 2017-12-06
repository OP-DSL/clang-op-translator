#ifndef PARLOOPHANDLER_H_INCLUDED
#define PARLOOPHANDLER_H_INCLUDED
#include "core/OPParLoopData.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"


namespace OP2 {
namespace matchers = clang::ast_matchers;

class ParLoopHandler : public matchers::MatchFinder::MatchCallback {
  std::map<std::string, clang::tooling::Replacements> *Replace;
  std::vector<ParLoop> &parLoops;

  void parseFunctionDecl(const clang::CallExpr *parloopExpr,
                         const clang::SourceManager *SM);

public:
  ParLoopHandler(std::map<std::string, clang::tooling::Replacements> *Replace,
                 std::vector<ParLoop> &parLoops)
      : Replace(Replace), parLoops(parLoops) {}

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};
} // namespace OP2
#endif // end of PARLOOPHANDLER_H_INCLUDED guard
