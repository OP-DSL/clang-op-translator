#ifndef MASTERKERNELGENERATOR_HPP
#define MASTERKERNELGENERATOR_HPP
#include "../OP2WriteableRefactoringTool.hpp"
#include "OMPRefactoringTool.h"
#include "SeqRefactoringTool.h"
#include "VecRefactoringTool.h"
#include "handler.hpp"
#include <algorithm>

namespace OP2 {

const DeclarationMatcher globVarMatcher =
    varDecl(hasName("global_var")).bind("global_var");

namespace matchers = clang::ast_matchers;
class MasterKernelHandler : public matchers::MatchFinder::MatchCallback {
  const std::vector<std::string> &kernels;
  const std::set<op_global_const> &constants;

  std::map<std::string, clang::tooling::Replacements> *Replace;

public:
  MasterKernelHandler(
      std::vector<std::string> &kernels,
      const std::set<op_global_const> &constants,
      std::map<std::string, clang::tooling::Replacements> *Replace)
      : kernels(kernels), constants(constants), Replace(Replace) {}

  std::string generateFile() {
    std::string repl;
    llvm::raw_string_ostream os(repl);
    for (const op_global_const &c : constants) {
      os << "extern " << c.type << " " << c.name;
      if (c.size != 1)
        os << "[" << c.size << "]";
      os << ";\n";
    }
    os << "// user kernel files\n";
    for (const std::string &kernel : kernels) {
      os << "#include \"" << kernel << "\"\n";
    }
    return os.str();
  }

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override {
    HANDLER(clang::VarDecl, 11, "global_var",
            MasterKernelHandler::generateFile);
  }
};

template <typename KernelGeneratorType, typename Handler = MasterKernelHandler>
class MasterkernelGenerator : public OP2WriteableRefactoringTool {
protected:
  const std::vector<ParLoop> &loops;
  const std::set<op_global_const> &constants;
  std::string base_name;
  std::vector<std::string> commandLineArgs;
  std::vector<std::string> generatedFiles;

public:
  MasterkernelGenerator(
      const std::vector<ParLoop> &loops,
      const std::set<op_global_const> &consts, std::string base,
      std::string skeleton_name, std::vector<std::string> &args,
      clang::tooling::FixedCompilationDatabase &compilations,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : OP2WriteableRefactoringTool(
            compilations, {std::string(SKELETONS_DIR) + skeleton_name},
            PCHContainerOps),
        loops(loops), constants(consts), base_name(base),
        commandLineArgs(args) {}

  MasterkernelGenerator(
      const std::vector<ParLoop> &loops,
      const std::set<op_global_const> &consts, std::string base,
      std::vector<std::string> &commandLineArgs,
      clang::tooling::FixedCompilationDatabase &compilations,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : MasterkernelGenerator(loops, consts, base, "skeleton_kernels.cpp",
                              commandLineArgs, compilations, PCHContainerOps) {}
  /// @brief Generates kernelfiles for all parLoop and then the master kernel
  ///  file
  void generateKernelFiles() {
    for (size_t i = 0; i < KernelGeneratorType::numParams; ++i) {
      commandLineArgs.push_back(KernelGeneratorType::commandlineParams[i]);
    }
    /* Here we could add the op_lib_cpp.h for kernel gen
     * but it gives errors in userkernel modifications..
    commandLineArgs.push_back(std::string("-include") + OP2_INC +
                              "op_lib_cpp.h");
    */
    clang::tooling::FixedCompilationDatabase F(".", commandLineArgs);
    for (const ParLoop &loop : loops) {
      std::string name = loop.getName();
      KernelGeneratorType tool(F, loop);
      if (tool.generateKernelFile()) {
        llvm::outs() << "Error during processing ";
      }
      llvm::outs() << name << "\n";
      generatedFiles.push_back(tool.getOutputFileName());
    }

    clang::ast_matchers::MatchFinder Finder;
    Handler handler(generatedFiles, constants, &getReplacements());
    Finder.addMatcher(globVarMatcher, &handler);
    if (int Result =
            run(clang::tooling::newFrontendActionFactory(&Finder).get())) {
      llvm::outs() << "Error " << Result << "\n";
      return;
    }
    writeOutReplacements();
  }

  std::string getOutputFileName(const clang::FileEntry *) const {
    return base_name + "_" + KernelGeneratorType::_postfix + "s.cpp";
  }
};

typedef MasterkernelGenerator<SeqRefactoringTool> SeqGenerator;
typedef MasterkernelGenerator<OMPRefactoringTool> OpenMPGenerator;
typedef MasterkernelGenerator<VecRefactoringTool> VectorizedGenerator;
} // namespace OP2

#endif /* ifndef MASTERKERNELGENERATOR_HPP */
