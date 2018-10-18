#include "AppFileRefactoringTool.hpp"
#include "AppFileTransformations.hpp"
#include "core/utils.h"
namespace OP2 {
std::string
AppFileRefactoringTool::getOutputFileName(const clang::FileEntry *Entry) const {
  llvm::outs() << "Rewrite buffer for file: " << Entry->getName() << "\n";

  std::string filename = Entry->getName().str();
  size_t basename_start = filename.rfind("/") + 1,
         basename_end = filename.rfind(".");
  if (basename_start == std::string::npos)
    basename_start = 0;
  if (basename_end == std::string::npos || basename_end < basename_start)
    llvm::errs() << "Invalid filename: " << Entry->getName() << "\n";
  filename = filename.substr(basename_start, basename_end - basename_start) +
             "_op.cpp";
  return filename;
}

int AppFileRefactoringTool::generateOPFiles() {
  using namespace clang::ast_matchers;
  clang::ast_matchers::MatchFinder Finder;
  ParLoopDeclarator callback(*this);
  auto loopHandler =
      make_matcher(ParloopCallReplaceOperation(application, callback, *this));

  Finder.addMatcher(
      callExpr(callee(functionDecl(hasName("op_par_loop")))).bind("par_loop"),
      &loopHandler);
  Finder.addMatcher(
      callExpr(callee(functionDecl(hasName("ops_par_loop")))).bind("par_loop"),
      &loopHandler);

  return run(
      clang::tooling::newFrontendActionFactory(&Finder, &callback).get());
}

AppFileRefactoringTool::AppFileRefactoringTool(
    clang::tooling::CommonOptionsParser &optionsParser, OPApplication &app)
    : OP2WriteableRefactoringTool(optionsParser), application(app) {
  // Add clang system headers to command line
  appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
      std::string("-isystem" + std::string(CLANG_SYSTEM_HEADERS) + "/include")
          .c_str()));
}

} // namespace OP2
