//------------------------------
#include "OP2RefactoringTool.hpp"
//-----------------------------

static llvm::cl::OptionCategory Op2Category("OP2 Options");
static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);
static llvm::cl::opt<OP2::OP2Targets>
    opTarget("optarget",
             llvm::cl::desc("Available versions to be generated by OP2:"),
             llvm::cl::init(OP2::all), llvm::cl::cat(Op2Category),
             llvm::cl::values(
                 clEnumValN(OP2::all, "all", "All possible versions [default]"),
                 clEnumValN(OP2::none, "none", "Don't generate kernel files"),
                 clEnumValN(OP2::seq, "seq", "Sequential"),
                 clEnumValN(OP2::openmp, "openmp", "OpenMP"),
                 clEnumValN(OP2::vec, "vec", "Vectorization")));

int main(int argc, const char **argv) {
  using namespace clang::tooling;
  using namespace clang::ast_matchers;
  CommonOptionsParser OptionsParser(argc, argv, Op2Category);

  std::vector<std::string> args = OP2::getCommandlineArgs(OptionsParser);
  args.insert(args.begin(), std::string("-I") + OP2_INC);
  clang::tooling::FixedCompilationDatabase Compilations(".", args);
  OP2::OP2RefactoringTool Tool(args, Compilations, OptionsParser,
                               opTarget.getValue());

  if (int Result = Tool.generateOPFiles()) {
    return Result;
  }

  Tool.generateKernelFiles();
  Tool.writeOutReplacements();

  return 0;
}