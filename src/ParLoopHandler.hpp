#ifndef PARLOOPHANDLER_H_INCLUDED
#define PARLOOPHANDLER_H_INCLUDED
#include "OPParLoopDeclarator.hpp"
#include "core/OPParLoopData.h"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Refactoring.h>

namespace OP2 {
class OP2RefactoringTool;
namespace matchers = clang::ast_matchers;

class ParLoopHandler : public matchers::MatchFinder::MatchCallback {
  OP2RefactoringTool &tool;
  OP2Application &app;
  OPParLoopDeclarator &declarator;

  void parseFunctionDecl(const clang::CallExpr *parloopExpr,
                         const clang::SourceManager *SM);

public:
  ParLoopHandler(OP2RefactoringTool &tool, OP2Application &app,
                 OPParLoopDeclarator &declarator);

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};
} // namespace OP2
#endif // end of PARLOOPHANDLER_H_INCLUDED guard
