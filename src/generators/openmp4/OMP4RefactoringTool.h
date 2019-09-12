#ifndef OMP4RefactoringTool_H
#define OMP4RefactoringTool_H

#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "generators/common/GeneratorBase.hpp"
#include "generators/openmp4/OMP4KernelHandler.h"
#include "generators/sequential/SeqKernelHandler.h"

namespace OP2 {
/// \brief Utility of generate OpenMP kernel based on ParLoop information.
///
///
class OMP4RefactoringTool : public OP2KernelGeneratorBase {
  static const std::string skeletons[2];

  /// @brief Handler for OpenMP kernel specific modifications.
  ///
  OMP4KernelHandler omp4KernelHandler;
  /// @brief Handler for modifications same as sequential case.
  ///
  SeqKernelHandler seqKernelHandler;



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
                               app, idx, OMP4RefactoringTool::_postfix, op),
        omp4KernelHandler(Compilations, &getReplacements(), app.getParLoops()[idx], app, idx),
        seqKernelHandler(&getReplacements(), app, idx) {}

  /// @brief Adding OpenMP specific Matchers and handlers.
  ///   Called from OP2KernelGeneratorBase::GenerateKernelFile()
  ///
  /// @param MatchFinder used by the RefactoringTool
  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &Finder) {
    Finder.addMatcher(OMPKernelHandler::locRedVarMatcher, &omp4KernelHandler);
    Finder.addMatcher(OMPKernelHandler::locRedToArgMatcher, &omp4KernelHandler);
    Finder.addMatcher(OMPKernelHandler::ompParForMatcher, &omp4KernelHandler);

    Finder.addMatcher(OMP4KernelHandler::userFuncMatcher, &omp4KernelHandler);
    Finder.addMatcher(OMP4KernelHandler::funcCallMatcher, &omp4KernelHandler);
    Finder.addMatcher(OMP4KernelHandler::mapIdxDeclMatcher, &omp4KernelHandler);
  }

  static constexpr const char *_postfix = "kernel";
  static constexpr const char *fileExtension = ".cpp";
  static constexpr unsigned numParams = 1;
  static constexpr const char *commandlineParams[numParams] = {"-fopenmp"};

  virtual ~OMP4RefactoringTool() = default;
};

const std::string OMP4RefactoringTool::skeletons[2] = {
    "skeleton_direct_OMP4_kernel.cpp", "skeleton_OMP4_kernels.cpp"};

constexpr const char *OMP4RefactoringTool::commandlineParams[OMP4RefactoringTool::numParams];

} // namespace OP2

#endif
