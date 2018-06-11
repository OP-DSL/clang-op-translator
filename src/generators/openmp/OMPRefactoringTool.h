#ifndef OMPREFACTORINGTOOL_H
#define OMPREFACTORINGTOOL_H

#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "generators/common/GeneratorBase.hpp"
#include "generators/openmp/OMPKernelHandler.h"
#include "generators/sequential/SeqKernelHandler.h"

namespace OP2 {
/// \brief Utility of generate OpenMP kernel based on ParLoop information.
///
///
class OMPRefactoringTool : public OP2KernelGeneratorBase {
  static const std::string skeletons[2];

  /// @brief Handler for OpenMP kernel specific modifications.
  ///
  OMPKernelHandler ompKernelHandler;
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
  OMPRefactoringTool(const clang::tooling::CompilationDatabase &Compilations,
                     const OP2Application &app, size_t idx, Staging)
      : OP2KernelGeneratorBase(Compilations,
                               {std::string(SKELETONS_DIR) +
                                skeletons[!app.getParLoops()[idx].isDirect()]},
                               app, idx, OMPRefactoringTool::_postfix),
        ompKernelHandler(&getReplacements(), app.getParLoops()[idx]),
        seqKernelHandler(&getReplacements(), app, idx) {}

  /// @brief Adding OpenMP specific Matchers and handlers.
  ///   Called from OP2KernelGeneratorBase::GenerateKernelFile()
  ///
  /// @param MatchFinder used by the RefactoringTool
  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &Finder) {
    Finder.addMatcher(OMPKernelHandler::locRedVarMatcher, &ompKernelHandler);
    Finder.addMatcher(OMPKernelHandler::locRedToArgMatcher, &ompKernelHandler);
    Finder.addMatcher(OMPKernelHandler::ompParForMatcher, &ompKernelHandler);
    Finder.addMatcher(SeqKernelHandler::userFuncMatcher, &seqKernelHandler);
    Finder.addMatcher(SeqKernelHandler::funcCallMatcher, &ompKernelHandler);
    Finder.addMatcher(SeqKernelHandler::mapIdxDeclMatcher, &seqKernelHandler);
  }

  static constexpr const char *_postfix = "kernel";
  static constexpr const char *fileExtension = ".cpp";
  static constexpr unsigned numParams = 1;
  static constexpr const char *commandlineParams[numParams] = {"-fopenmp"};

  virtual ~OMPRefactoringTool() = default;
};

const std::string OMPRefactoringTool::skeletons[2] = {
    "skeleton_direct_kernel.cpp", "skeleton_kernel.cpp"};

} // namespace OP2

#endif
