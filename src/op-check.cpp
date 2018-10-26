#include "op-check.hpp"
#include "core/utils.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include <functional>
#include <vector>

static llvm::cl::OptionCategory opCheckCategory("OP Check Options");
static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

int main(int argc, const char **argv) {
  clang::tooling::CommonOptionsParser OptionsParser(argc, argv,
                                                    opCheckCategory);
  OP2::OPApplication application;

  OP2::CheckTool Tool(OptionsParser, application);
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
