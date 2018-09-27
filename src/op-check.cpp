#include "op-check.hpp"
#include "core/utils.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include <functional>
#include <vector>
using namespace clang;

static llvm::cl::OptionCategory opCheckCategory("OP Check Options");
static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

int main(int argc, const char **argv) {
  using namespace clang::ast_matchers;

  clang::tooling::CommonOptionsParser OptionsParser(argc, argv,
                                                    opCheckCategory);

  OP2::OPCheckTool Tool(OptionsParser);
  Tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
      std::string("-isystem" + std::string(CLANG_SYSTEM_HEADERS) + "/include")
          .c_str()));
  return Tool.setFinderAndRun();
}
