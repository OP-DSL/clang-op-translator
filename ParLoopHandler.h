#ifndef __PARLOOPHANDLER_H_INCLUDED__
#define  __PARLOOPHANDLER_H_INCLUDED__
#include "clang/Tooling/Refactoring.h"
#include "op_par_loop.h"
//TODO check includes
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace clang {
class CompilerInstance;
}


namespace OP2 {
namespace matchers = clang::ast_matchers;

class ParLoopHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
  std::map<std::string, clang::tooling::Replacements> *Replace;
  std::vector<ParLoop>& parLoops;

  void  parseFunctionDecl(const clang::CallExpr * parloopExpr,
                          const clang::SourceManager *SM);

public:
  ParLoopHandler(std::map<std::string, clang::tooling::Replacements> *Replace,
                 std::vector<ParLoop>& parLoops)
	  : Replace(Replace), parLoops(parLoops) {}

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;

};
}
#endif //end of __PARLOOPHANDLER_H_INCLUDED__ guard
