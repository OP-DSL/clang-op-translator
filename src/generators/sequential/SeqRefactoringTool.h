#ifndef SEQREFACTORINGTOOL_H
#define SEQREFACTORINGTOOL_H

#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "generators/common/GeneratorBase.hpp"
#include "generators/sequential/SeqKernelHandler.h"
namespace OP2 {
/// @brief Utility of generate sequential kernel based on ParLoop information.
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
  /// @param app collected application data.
  /// @param idx idx of the currently processed parloop.
  SeqRefactoringTool(const clang::tooling::CompilationDatabase &Compilations,
                     const OP2Application &app, size_t idx, Staging)
      : OP2KernelGeneratorBase(Compilations,
                               {std::string(SKELETONS_DIR) + skeletons[0]}, app,
                               idx, SeqRefactoringTool::_postfix),
        seqKernelHandler(&getReplacements(), app, idx) {}

  /// @brief Adding Sequential specific MAtchers and handlers.
  ///   Called from OP2KernelGeneratorBase::GenerateKernelFile()
  ///
  /// @param MatchFinder used by the RefactoringTool
  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &Finder) {
    Finder.addMatcher(SeqKernelHandler::userFuncMatcher, &seqKernelHandler);
    Finder.addMatcher(SeqKernelHandler::funcCallMatcher, &seqKernelHandler);
    Finder.addMatcher(SeqKernelHandler::mapIdxDeclMatcher, &seqKernelHandler);
  }

  static constexpr const char *_postfix = "seqkernel";
  static constexpr const char *fileExtension = ".cpp";
  static constexpr unsigned numParams = 0;
  static constexpr const char *commandlineParams[numParams] = {};

  virtual ~SeqRefactoringTool() = default;
};

} // namespace OP2

#endif
