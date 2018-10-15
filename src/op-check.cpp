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
  OP2::OPApplication application;

  OP2::OPCheckTool Tool(OptionsParser, application);
  Tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
      std::string("-isystem" + std::string(CLANG_SYSTEM_HEADERS) + "/include")
          .c_str()));
  if (int err = Tool.setFinderAndRun()) {
    return err;
  }
#ifndef NDEBUG
  OP2::debugs() << application.applicationName
                << "\nList of application files: \n";
  for (const auto &fname : application.applicationFiles) {
    OP2::debugs() << "\t* " << fname << '\n';
  }
  OP2::debugs() << "\nList of defined global constants:\n";
  for (const auto &c : application.constants) {
    OP2::debugs() << "\t* " << c << '\n';
  }
  OP2::debugs() << "\nList of par_loops found:\n";
  for (const auto &loop : application.loops) {
    loop.prettyPrint(OP2::debugs());
    OP2::debugs() << "\n";
  }
#endif
  return 0;
}
