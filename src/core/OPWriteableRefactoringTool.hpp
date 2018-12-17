#ifndef OPWRITEABLEREFACTORINGTOOL_HPP
#define OPWRITEABLEREFACTORINGTOOL_HPP

#include "core/utils.h"
// clang Tooling includes
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"

namespace op_dsl {
/// @brief Base class for RefactoringTools in op2-clang. Adds utility to write
/// output to files or streams.
class OPWriteableRefactoringTool : public clang::tooling::RefactoringTool {
public:
  /// @brief Generate output filename.
  ///
  /// @param Entry The input file that processed
  ///
  /// @return output filename (e.g. xxx_op.cpp, xxx_kernel.cpp)
  virtual std::string
  getOutputFileName(const clang::FileEntry *Entry) const = 0;

  explicit OPWriteableRefactoringTool(
      clang::tooling::CommonOptionsParser &optionsParser)
      : clang::tooling::RefactoringTool(optionsParser.getCompilations(),
                                        optionsParser.getSourcePathList()) {}
  OPWriteableRefactoringTool(
      const clang::tooling::CompilationDatabase &Compilations,
      const std::vector<std::string> &Sources)
      : clang::tooling::RefactoringTool(Compilations, Sources) {}

  /// @brief Create the output files based on the replacements.
  virtual void writeOutReplacements() {
    writeReplacementsTo(
        [this](const clang::FileEntry *Entry) {
          return getOutputFileName(Entry);
        },
        this);
  }

  void addReplacementTo(const std::string &fileName,
                        const clang::tooling::Replacement &repl,
                        const std::string &diagMessage) {
    llvm::Error err = getReplacements()[fileName].add(repl);
    if (err) { // TODO(bgd54): proper error checking
      llvm::outs() << "Some Error occured during adding replacement for "
                   << diagMessage << "\n";
    }
  }

  virtual ~OPWriteableRefactoringTool() = default;
  OPWriteableRefactoringTool(const OPWriteableRefactoringTool &) = delete;
  OPWriteableRefactoringTool(const OPWriteableRefactoringTool &&) = delete;
  OPWriteableRefactoringTool &
  operator=(const OPWriteableRefactoringTool &) = delete;
  OPWriteableRefactoringTool &
  operator=(const OPWriteableRefactoringTool &&) = delete;
};

} // namespace op_dsl

#endif /* ifndef OPWRITEABLEREFACTORINGTOOL_HPP */
