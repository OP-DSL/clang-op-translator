#ifndef MASTERKERNELGENERATOR_HPP
#define MASTERKERNELGENERATOR_HPP
#include "../OP2WriteableRefactoringTool.hpp"
#include "OMPRefactoringTool.h"
#include "SeqRefactoringTool.h"

namespace OP2 {

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
      : OP2WriteableRefactoringTool(optionsParser.getCompilations(), {/*TODO*/},
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
    //    Finder.addMatcher();
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
