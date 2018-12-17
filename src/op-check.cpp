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
  op_dsl::OPApplication application;

  op_dsl::CheckTool Tool(OptionsParser, application);
  if (int err = Tool.setFinderAndRun()) {
    return err;
  }
#ifndef NDEBUG
  op_dsl::debugs() << application.applicationName
                   << "\nList of application files: \n";
  for (const auto &fname : application.applicationFiles) {
    op_dsl::debugs() << "\t* " << fname << '\n';
  }
  op_dsl::debugs() << "\nList of defined global constants:\n";
  for (const auto &c : application.constants) {
    op_dsl::debugs() << "\t* " << c << '\n';
  }
  op_dsl::debugs() << "\nList of par_loops found:\n";
  for (const auto &loop : application.loops) {
    loop.prettyPrint(op_dsl::debugs());
    op_dsl::debugs() << "\n";
  }
#endif
  return 0;
}
