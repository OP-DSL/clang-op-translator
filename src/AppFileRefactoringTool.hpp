#ifndef OPAPPFILEREFACTORINGTOOL_HPP
#define OPAPPFILEREFACTORINGTOOL_HPP
#include "core/OPParLoopData.h"
#include "core/OPWriteableRefactoringTool.hpp"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace op_dsl {

/**
 * @brief Utility to transform the apllication files of an OP2 or OPS
 * application.
 *
 * With generateOPFiles this tool runs through a set of matcher on the
 * application files and generates replacements for par_loop calls and constant
 * declarations. Then with writeOutReplacements the generates the modified
 * application files (xxx_op.cpp xxx_ops.cpp).
 */
class AppFileRefactoringTool final : public OPWriteableRefactoringTool {
public:
  AppFileRefactoringTool(clang::tooling::CommonOptionsParser &optionsParser,
                         OPApplication &app);

  /// @brief Setting the finders for the refactoring tool then runs the tool
  ///   to generate the replacements for the application files.
  ///
  /// @return 0 on success
  int generateOPFiles();

  std::string getOutputFileName(const clang::FileEntry *Entry) const override;

private:
  OPApplication &application;
};

} // namespace op_dsl
#endif /* OPAPPFILEREFACTORINGTOOL_HPP */
