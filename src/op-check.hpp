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
class CheckTool : public clang::tooling::ClangTool {
private:
  OPApplication &application; // TODO(bgd54): think about making this a
                              // unique_ptr if we need only the checks

public:
  CheckTool(clang::tooling::CommonOptionsParser &optionsParser,
            OPApplication &app)
      : ClangTool(optionsParser.getCompilations(),
                  optionsParser.getSourcePathList()),
        application(app) {
    // Initialize some data on application.
    std::string applicationName = optionsParser.getSourcePathList()[0];
    size_t basename_start = applicationName.rfind('/'),
           basename_end = applicationName.rfind('.');
    if (basename_start == std::string::npos) {
      basename_start = 0;
    } else {
      basename_start++;
    }
    if (basename_end == std::string::npos || basename_end < basename_start) {
      llvm::errs() << "Invalid applicationName: " << applicationName << "\n";
    }
    applicationName =
        applicationName.substr(basename_start, basename_end - basename_start);
    application.setName(applicationName);
    application.applicationFiles =
        optionsParser.getSourcePathList(); // TODO(bgd54): absolute paths --
                                           // beginsourcefile action or sg.
    // Add clang system headers to command line
    appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
        std::string("-isystem" + std::string(CLANG_SYSTEM_HEADERS) + "/include")
            .c_str()));
  }

  int setFinderAndRun() {
    using namespace clang::ast_matchers;
    clang::ast_matchers::MatchFinder finder;
    MatchMaker<CheckSingleCallOperation> m("op_init or ops_init");
    MatchMaker<CheckSingleCallOperation> m2("op_exit or ops_exit");
    auto parLoopParser = make_matcher(ParLoopParser(application));
    auto constRegister = make_matcher(OPConstRegister(application));
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
    finder.addMatcher(callExpr(callee(functionDecl(hasName("ops_par_loop"))))
                          .bind("par_loop"),
                      &parLoopParser);
    finder.addMatcher(callExpr(callee(functionDecl(hasName("op_decl_const"))))
                          .bind("decl_const"),
                      &constRegister);
    finder.addMatcher(callExpr(callee(functionDecl(hasName("ops_decl_const"))))
                          .bind("decl_const"),
                      &constRegister);
    return run(clang::tooling::newFrontendActionFactory(&finder).get());
  }
};

} // namespace OP2
#endif /* OP_CHECK_HPP */
