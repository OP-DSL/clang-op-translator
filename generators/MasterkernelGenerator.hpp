#ifndef MASTERKERNELGENERATOR_HPP
#define MASTERKERNELGENERATOR_HPP
#include "../OP2WriteableRefactoringTool.hpp"
#include "OMPRefactoringTool.h"
#include "SeqRefactoringTool.h"
#include "handler.hpp"

namespace OP2 {

const DeclarationMatcher globVarMatcher =
    varDecl(hasName("global_var")).bind("global_var");

namespace matchers = clang::ast_matchers;
class MasterKernelHandler : public matchers::MatchFinder::MatchCallback {
  std::vector<std::string> &kernels;
  std::map<std::string, clang::tooling::Replacements> *Replace;

public:
  MasterKernelHandler(
      std::vector<std::string> &kernels,
          std::map<std::string, clang::tooling::Replacements> *Replace)
      : kernels(kernels), Replace(Replace) {}

  std::string generateFile() {
    std::string repl = "// user kernel files\n";
    llvm::raw_string_ostream os(repl);
    for (std::string &kernel : kernels) {
      os << "#include \"" << kernel << "\"\n";
    }
    return os.str();
  }

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override {
    HANDLER(clang::VarDecl, 11, "global_var",
            MasterKernelHandler::generateFile);
  }
};

template <typename KernelGeneratorType>
class MasterkernelGenerator : public OP2WriteableRefactoringTool {
protected:
  const std::vector<ParLoop> &loops;
  std::string base_name;
  clang::tooling::CommonOptionsParser &optionsParser;
  std::vector<std::string> generatedFiles;

public:
  MasterkernelGenerator(
      const std::vector<ParLoop> &loops, std::string base,
      clang::tooling::CommonOptionsParser &optionsParser,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : OP2WriteableRefactoringTool(
            optionsParser.getCompilations(),
            {std::string(SKELETONS_DIR) + "skeleton_kernels.cpp"},
            PCHContainerOps),
        loops(loops), base_name(base), optionsParser(optionsParser) {}

  /// @brief Generates kernelfiles for all parLoop
  void generateKernelFiles() {
    for (const ParLoop &loop : loops) {
      std::string name = loop.getName();
      KernelGeneratorType tool(optionsParser.getCompilations(), loop);
      if (tool.generateKernelFile()) {
        llvm::outs() << "Error during processing ";
      }
      llvm::outs() << name << "\n";
      generatedFiles.push_back(tool.getOutputFileName());
    }

    clang::ast_matchers::MatchFinder Finder;
    MasterKernelHandler handler(generatedFiles, &getReplacements());
    Finder.addMatcher(globVarMatcher, &handler);
    if (int Result =
            run(clang::tooling::newFrontendActionFactory(&Finder).get())) {
      llvm::outs() << "Error " << Result << "\n";
      return;
    }
    writeOutReplacements();
  }

  std::string getOutputFileName(const clang::FileEntry *Entry) const {
    return base_name + "_" + KernelGeneratorType::_postfix + "s.cpp";
  }
};

typedef MasterkernelGenerator<SeqRefactoringTool> SeqGenerator;
typedef MasterkernelGenerator<OMPRefactoringTool> OpenMPGenerator;
} // namespace OP2

#endif /* ifndef MASTERKERNELGENERATOR_HPP */
