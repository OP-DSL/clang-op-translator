#pragma once
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

class ParLoopHandler : public clang::ast_matchers::MatchFinder::MatchCallback {

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
private:
  static void arg_dat_processor(const matchers::MatchFinder::MatchResult &Result);
  static void arg_gbl_processor(const matchers::MatchFinder::MatchResult &Result);
};
}
