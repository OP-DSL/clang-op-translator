#ifndef OP_CHECK_HPP
#define OP_CHECK_HPP
#include "checkTransformations.hpp"
#include "core/utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include <functional>

namespace OP2 {

/**
 * @brief Utility to run basic check on OP2 and OPS application files.
 *
 * This tool perform basic checks on OP2 or OPS application files and create the
 * OPApplication object for code generation passes.
 */
class OPCheckTool : public clang::tooling::ClangTool {
private:
  OPApplication application;

public:
  OPCheckTool(clang::tooling::CommonOptionsParser &optionsParser)
      : ClangTool(optionsParser.getCompilations(),
                  optionsParser.getSourcePathList()) {}

  int setFinderAndRun() {
    using namespace clang;
    using namespace ast_matchers;
    clang::ast_matchers::MatchFinder finder;
    MatchMaker<CheckSingleCallOperation> m("op_init or ops_init");
    MatchMaker<CheckSingleCallOperation> m2("op_exit or ops_exit");
    auto parLoopParser =
        make_matcher<ParLoopParser>(ParLoopParser(application));
    finder.addMatcher(
        callExpr(callee(functionDecl(hasName("op_init")))).bind("callExpr"),
        &m);
    finder.addMatcher(
        callExpr(callee(functionDecl(hasName("ops_init")))).bind("callExpr"),
        &m);
    finder.addMatcher(
        callExpr(callee(functionDecl(hasName("ops_exit")))).bind("callExpr"),
        &m2);
    finder.addMatcher(
        callExpr(callee(functionDecl(hasName("op_exit")))).bind("callExpr"),
        &m2);
    finder.addMatcher(
        callExpr(callee(functionDecl(hasName("op_par_loop")))).bind("par_loop"),
        &parLoopParser);
    return run(clang::tooling::newFrontendActionFactory(&finder).get());
  }
};

} // namespace OP2
#endif /* OP_CHECK_HPP */
