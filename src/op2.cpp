//------------------------------
#include "AppFileRefactoringTool.hpp"
#include "core/op2_clang_core.h"
#include "core/utils.h"
#include "op-check.hpp"
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
                 clEnumValN(OP2::cuda, "cuda", "CUDA"),
                 clEnumValN(OP2::vec, "vec", "Vectorization")));
static llvm::cl::opt<OP2::Staging> staging(
    "staging", llvm::cl::desc("Sets the staging and coloring type for cuda."),
    llvm::cl::init(OP2::OP_STAGE_ALL), llvm::cl::cat(Op2Category),
    llvm::cl::values(
        clEnumValN(OP2::OP_STAGE_ALL, "op_stage_all",
                   "Use hierarchical coloring with staging [default]"),
        clEnumValN(OP2::OP_COlOR2, "op_color2", "Use global coloring")));
static llvm::cl::opt<bool>
    SOA("soa",
        llvm::cl::desc("Enable AoS to SoA transformation default: false"),
        llvm::cl::init(false), llvm::cl::cat(Op2Category));

int main(int argc, const char **argv) {
  using clang::tooling::CommonOptionsParser;
  // Initialize the tool
  CommonOptionsParser OptionsParser(argc, argv, Op2Category);

  OP2::OPApplication application;
  // TODO add another way to initialize the application
  // Parse the application files, build model
  if (OP2::CheckTool parserTool(OptionsParser, application);
      int err = parserTool.setFinderAndRun()) {
    return err;
  }

  // TODO sourcepathlist from app..
  OP2::AppFileRefactoringTool applicationRefactor(OptionsParser, application);
  if (int err = applicationRefactor.generateOPFiles()) {
    return err;
  }
  applicationRefactor.writeOutReplacements();

  //___________________
  /*
  OP2::OP2Optimizations optim{staging.getValue(), SOA.getValue()};
  std::vector<std::string> args = OP2::getCommandlineArgs(OptionsParser);
  args.insert(args.begin(), std::string("-I") + OP2_INC);
  clang::tooling::FixedCompilationDatabase Compilations(".", args);

  OP2::OP2RefactoringTool Tool(args, Compilations, OptionsParser,
                               opTarget.getValue(), optim);
  // Collect data about the application and generate modified application files.
  if (int Result = Tool.generateOPFiles()) { return Result; }

  // Generate and write target specific kernel files.
  Tool.generateKernelFiles();
  // Write out the modified application files.
  */
  return 0;
}
