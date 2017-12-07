#ifndef PARLOOPHANDLER_H_INCLUDED
#define PARLOOPHANDLER_H_INCLUDED
#include "core/OPParLoopData.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

class ParLoopHandler : public matchers::MatchFinder::MatchCallback {
  std::map<std::string, clang::tooling::Replacements> *Replace;
  OP2Application &app;

  void parseFunctionDecl(const clang::CallExpr *parloopExpr,
                         const clang::SourceManager *SM);

public:
  ParLoopHandler(std::map<std::string, clang::tooling::Replacements> *Replace,
                 OP2Application &app)
      : Replace(Replace), app(app) {}

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};
} // namespace OP2
#endif // end of PARLOOPHANDLER_H_INCLUDED guard
