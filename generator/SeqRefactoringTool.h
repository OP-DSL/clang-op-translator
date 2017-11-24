#ifndef SEQREFACTORINGTOOL_H
#define SEQREFACTORINGTOOL_H

#include "../OPParLoopData.h"
#include "GeneratorBase.hpp"
#include "SeqKernelHandler.h"
namespace OP2 {
/// \brief Utility of generate sequential kernel based on ParLoop information.
///
///
class SeqRefactoringTool : public OP2KernelGeneratorBase {
  static constexpr const char *skeletons[1] = {"skeleton_seqkernel.cpp"};

  /// @brief Handler for Sequential kernel specific modifications.
  ///
  SeqKernelHandler seqKernelHandler;

public:
  /// @brief Construct a refactoring tool to generate the sequential kernel.
  ///
  /// @param Compilations The CompilationDatabase which contains the copmile
  /// commandlines.
  /// @param loop The ParLoop containing informations about the op_par_loop.
  /// @param PCHContainerOps The PCHContainerOperation for loading and creating
  /// clang modules
  // TODO: Modify to get right skeletons... and Database..
  SeqRefactoringTool(
      const clang::tooling::CompilationDatabase &Compilations,
      const ParLoop &loop,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>());

  /// @brief Adding Sequential specific MAtchers and handlers.
  ///   Called from OP2KernelGeneratorBase::GenerateKernelFile()
  ///
  /// @param MatchFinder used by the RefactoringTool
  virtual void addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &);

  static constexpr const char *_postfix = "seqkernel";

  virtual ~SeqRefactoringTool() = default;
};

} // namespace OP2

#endif
