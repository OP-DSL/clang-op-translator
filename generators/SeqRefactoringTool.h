#ifndef SEQREFACTORINGTOOL_H
#define SEQREFACTORINGTOOL_H

#include "../op_par_loop.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

namespace OP2 {
/// \brief Utility of generate sequential kernel based on ParLoop information.
///
///
class SeqRefactoringTool : public clang::tooling::RefactoringTool {
  //TODO maybe subclass of OP2RefactoringTool?
  const ParLoop &loop;

public:
  /// @brief Construct a refactoring tool to generate the sequential kernel.
  ///
  /// @param Compilations The CompilationDatabase which contains the copmile
  /// commandlines.
  /// @param loop The ParLoop containing informations about the op_par_loop.
  /// @param PCHContainerOps The PCHContainerOperation for loading and creating
  /// clang modules
  //TODO: Modify to get right skeletons... and Database..
  SeqRefactoringTool(
      const clang::tooling::CompilationDatabase &Compilations,
      const ParLoop &loop,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>());

  /// @brief Denerate the kernel to <loopname>_seqkernel.cpp
  ///
  /// @return 0 on success
  int generateKernelFile();
};

} // namespace OP2

#endif
