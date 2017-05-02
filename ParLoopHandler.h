#pragma once
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace clang {
class CompilerInstance;
}


namespace OP2 {
namespace matchers = clang::ast_matchers;

class ParLoopHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
  clang::Rewriter &Rewriter;

public:
  ParLoopHandler(clang::Rewriter &Rewriter) : Rewriter{Rewriter} {}

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;

};
}
