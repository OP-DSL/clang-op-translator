#include "ParLoopHandler.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include <memory>
#include <sstream>

namespace OP2 {

class ASTConsumer : public clang::ASTConsumer {
  clang::ast_matchers::MatchFinder Matcher;
  ParLoopHandler LoopHandler;
public:
  ASTConsumer(clang::Rewriter &R) : LoopHandler{R} {
    using namespace clang::ast_matchers;
    Matcher.addMatcher(callExpr(callee(functionDecl(hasName("op_par_loop")))).bind("par_loop"), &LoopHandler);
  }

  void HandleTranslationUnit(clang::ASTContext& context) override {
    Matcher.matchAST(context);
  }
};

class FrontendAction : public clang::ASTFrontendAction {

  clang::Rewriter Rewriter;

public:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& Compiler,
                                                        llvm::StringRef file) override {
    Rewriter.setSourceMgr(Compiler.getSourceManager(), Compiler.getLangOpts());
    return llvm::make_unique<ASTConsumer>(Rewriter);
  }

  void EndSourceFileAction() override {
    Rewriter.getEditBuffer(Rewriter.getSourceMgr().getMainFileID())
        .write(llvm::outs());
  }
};

} // namespace OP2

static llvm::cl::OptionCategory Op2Category("OP2 Options");
static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

int main(int argc, const char** argv)
{
  using namespace clang::tooling;
  CommonOptionsParser OptionsParser(argc, argv, Op2Category);

  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
  Tool.appendArgumentsAdjuster(
      [](const clang::tooling::CommandLineArguments &args,
         llvm::StringRef filename) {
        std::string s = std::string("-I")
                            .append(std::getenv("OP2_INSTALL_PATH"))
                            .append("/c/include");
        auto new_args = args;
        new_args.push_back(s);
        return new_args;
      });

  return Tool.run(clang::tooling::newFrontendActionFactory<OP2::FrontendAction>().get());
}

