#ifndef OMP4REFACTORINGTOOL_H
#define OMP4REFACTORINGTOOL_H

#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "generators/common/GeneratorBase.hpp"

namespace OP2 {
/// \brief Utility of generate OpenMP4 kernel based on ParLoop information.
///
///
class OMP4RefactoringTool : public OP2KernelGeneratorBase {
  static const std::string skeletons[2];


public:
  /// @brief Construct a refactoring tool to generate the OpenMP kernel.
  ///
  /// @param Compilations The CompilationDatabase which contains the copmile
  /// commandlines.
  /// @param loop The ParLoop containing informations about the op_par_loop.
  /// @param PCHContainerOps The PCHContainerOperation for loading and creating
  /// clang modules
  OMP4RefactoringTool(const clang::tooling::CompilationDatabase &Compilations,
                     const OP2Application &app, size_t idx, OP2Optimizations op)
      : OP2KernelGeneratorBase(Compilations,
                               {std::string(SKELETONS_DIR) +
                                skeletons[!app.getParLoops()[idx].isDirect()]},
                               app, idx, OMP4RefactoringTool::_postfix, op) {}

  /// @brief Adding OpenMP4 specific Matchers and handlers.
  ///   Called from OP2KernelGeneratorBase::GenerateKernelFile()
  ///
  /// @param MatchFinder used by the RefactoringTool
  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &Finder) {}

  static constexpr const char *_postfix = "omp4kernel";
  static constexpr const char *fileExtension = ".cpp";
  static constexpr unsigned numParams = 1;
  static constexpr const char *commandlineParams[numParams] = {"-fopenmp"};

  virtual ~OMP4RefactoringTool() = default;
};

const std::string OMP4RefactoringTool::skeletons[2] = {
    "skeleton_direct_kernel.cpp", "skeleton_kernel.cpp"};

} // namespace OP2

#endif /* ifndef OMP4REFACTORINGTOOL_H */
