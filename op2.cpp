//------------------------------
#include "OP2RefactoringTool.hpp"
//-----------------------------

static llvm::cl::OptionCategory Op2Category("OP2 Options");
static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

int main(int argc, const char **argv) {
  using namespace clang::tooling;
  using namespace clang::ast_matchers;
  CommonOptionsParser OptionsParser(argc, argv, Op2Category);

  OP2::OP2RefactoringTool Tool(OptionsParser);

  OP2::ParLoopHandler parLoopHandlerCallback(&Tool.getReplacements(),
                                             Tool.getParLoops());

  clang::ast_matchers::MatchFinder Finder;
  Finder.addMatcher(
      callExpr(callee(functionDecl(hasName("op_par_loop")))).bind("par_loop"),
      &parLoopHandlerCallback);

  if (int Result = Tool.run(newFrontendActionFactory(&Finder).get())) {
    return Result;
  }

  Tool.generateKernelFiles();
  Tool.writeOutReplacements();

  return 0;
}
