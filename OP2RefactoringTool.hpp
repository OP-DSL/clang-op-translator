#ifndef OP2REFACTORINGTOOL_HPP
#define OP2REFACTORINGTOOL_HPP

// OP2 includes
#include "OP2WriteableRefactoringTool.hpp"
#include "ParLoopHandler.h"
#include "generators/OMPRefactoringTool.h"
// Parloops.

namespace OP2 {

class OP2RefactoringTool : public OP2WriteableRefactoringTool {
protected:
  clang::tooling::CommonOptionsParser &optionsParser;
  // We can collect all data about kernels
  std::vector<ParLoop> loops;
  std::map<std::string, const op_set> sets;
  std::map<std::string, const op_map> mappings;

public:
  OP2RefactoringTool(
      clang::tooling::CommonOptionsParser &optionsParser,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : OP2WriteableRefactoringTool(optionsParser, PCHContainerOps),
        optionsParser(optionsParser) {}

  std::vector<ParLoop> &getParLoops() { return loops; }

  /// @brief Generates kernelfiles for all parLoop
  /// Currently only seqkernels created.
  void generateKernelFiles() {
    for (ParLoop &loop : loops) {
      std::string name = loop.getName();
      OMPRefactoringTool tool(optionsParser.getCompilations(), loop);
      if (tool.generateKernelFile()) {
        llvm::outs() << "Error during processing ";
      }
      llvm::outs() << name << "\n";
    }
  }

  /// @brief Generate output filename.
  ///
  /// @param Entry The input file that processed.
  ///
  /// @return output filename (xxx_op.cpp)
  virtual std::string getOutputFileName(const clang::FileEntry *Entry) const {
    llvm::outs() << "Rewrite buffer for file: " << Entry->getName() << "\n";

    std::string filename = Entry->getName().str();
    size_t basename_start = filename.rfind("/") + 1,
           basename_end = filename.rfind(".");
    if (basename_start == std::string::npos)
      basename_start = 0;
    if (basename_end == std::string::npos || basename_end < basename_start)
      llvm::errs() << "Invalid filename: " << Entry->getName() << "\n";
    filename = filename.substr(basename_start, basename_end - basename_start) +
               "_op.cpp";
    return filename;
  }
};

} // namespace OP2

#endif /* ifndef OP2REFACTORINGTOOL_HPP */
