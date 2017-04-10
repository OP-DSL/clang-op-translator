#include "llvm/Support/CommandLine.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <memory>
#include "ParLoopHandler.h"

namespace OP2 {

class ASTConsumer : public clang::ASTConsumer {
  clang::ast_matchers::MatchFinder Matcher;
  ParLoopHandler LoopHandler;
public:
  ASTConsumer()
  {
    using namespace clang::ast_matchers;
    Matcher.addMatcher(callExpr(callee(functionDecl(hasName("op_par_loop")))).bind("par_loop"), &LoopHandler);
  }

  void HandleTranslationUnit(clang::ASTContext& context) override {
    Matcher.matchAST(context);
  }
};

class FrontendAction : public clang::ASTFrontendAction {

public:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& Compiler,
                                                        llvm::StringRef file) override {
    return llvm::make_unique<ASTConsumer>();
  }
};

} // namespace OP2

static llvm::cl::OptionCategory Op2Category("Op2");


int main(int argc, const char** argv)
{
  using namespace clang::tooling;
  CommonOptionsParser OptionsParser(argc, argv, Op2Category);

  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

  return Tool.run(clang::tooling::newFrontendActionFactory<OP2::FrontendAction>().get());
}

