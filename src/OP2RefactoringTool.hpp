#ifndef OP2REFACTORINGTOOL_HPP
#define OP2REFACTORINGTOOL_HPP

// OP2 includes
#include "core/OP2WriteableRefactoringTool.hpp"
#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace OP2 {

/// @brief This class is responsible for processing OP2 applications.
/// Collect data about parloops and constants. Also generates the modified
/// application files.
///
class OP2RefactoringTool : public OP2WriteableRefactoringTool {
protected:
  OP2Targets opTarget;
  Staging staging;
  std::vector<std::string> &commandLineArgs;
  clang::tooling::FixedCompilationDatabase &Compilations;
  // We can collect all data about kernels
  OP2Application application;

public:
  OP2RefactoringTool(std::vector<std::string> &commandLineArgs,
                     clang::tooling::FixedCompilationDatabase &compilations,
                     clang::tooling::CommonOptionsParser &optionsParser,
                     OP2Targets opTarget, Staging staging);

  /// @brief Generates target specific kernelfiles for all parLoop and all
  /// specified target.
  void generateKernelFiles();

  /// @brief Setting the finders for the refactoring tool then runs the tool
  ///   to collect data about the op calls in the application and generate the
  ///   xxx_op files.
  ///
  /// @return 0 on success
  int generateOPFiles();

  /// @brief Generate output filename from the filename of the processed file.
  /// Inherited function from OP2WriteableRefactoringTool.
  ///
  /// @param Entry The input file that processed.
  ///
  /// @return output filename (xxx_op.cpp)
  virtual std::string getOutputFileName(const clang::FileEntry *Entry) const;
};

} // namespace OP2

#endif /* ifndef OP2REFACTORINGTOOL_HPP */
