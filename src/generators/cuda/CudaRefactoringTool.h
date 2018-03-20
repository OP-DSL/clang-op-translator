#ifndef CUDAREFACTORINGTOOL_H
#define CUDAREFACTORINGTOOL_H

#include "core/OPParLoopData.h"
#include "generators/common/GeneratorBase.hpp"
#include "generators/cuda/CudaKernelHandler.h"
#include "generators/sequential/SeqKernelHandler.h"

namespace OP2 {
/// \brief Utility of generate CUDA kernel based on ParLoop information.
///
///
class CUDARefactoringTool : public OP2KernelGeneratorBase {
  static const std::string skeletons[2];

  /// @brief Handler for CUDA specific modifications.
  ///
  CUDAKernelHandler cudaKernelHandler;

public:
  /// @brief Construct a refactoring tool to generate the CUDA kernel.
  ///
  /// @param Compilations The CompilationDatabase which contains the copmile
  /// commandlines.
  /// @param app Collected application data
  /// @param idx index of currently generated loop
  CUDARefactoringTool(const clang::tooling::CompilationDatabase &Compilations,
                      const OP2Application &app, size_t idx)
      : OP2KernelGeneratorBase(Compilations,
                               {std::string(SKELETONS_DIR) + skeletons[0]}, app,
                               idx, CUDARefactoringTool::_postfix),
        cudaKernelHandler(&getReplacements(), app, idx) {}

  /// @brief Adding CUDA specific Matchers and handlers.
  ///   Called from OP2KernelGeneratorBase::GenerateKernelFile()
  ///
  /// @param MatchFinder used by the RefactoringTool
  virtual void addGeneratorSpecificMatchers(
      clang::ast_matchers::MatchFinder &Finder) override {

    Finder.addMatcher(SeqKernelHandler::userFuncMatcher, &cudaKernelHandler);
    Finder.addMatcher(CUDAKernelHandler::cudaFuncMatcher, &cudaKernelHandler);
    Finder.addMatcher(CUDAKernelHandler::cudaFuncCallMatcher,
                      &cudaKernelHandler);
  }

  static constexpr const char *_postfix = "kernel";
  static constexpr unsigned numParams = 0;
  static const std::string commandlineParams[numParams];

  /// @brief Generate the name of the output file.
  /// Overrides function from OP2KernelGeneratorBase
  /// @param Entry The input file that processed
  ///
  /// @return output filename (xxx_kernel.cu)
  virtual std::string
  getOutputFileName(const clang::FileEntry *f = nullptr) const override {
    return application.getParLoops()[loopIdx].getName() + "_" + postfix + ".cu";
  }

  virtual ~CUDARefactoringTool() = default;
};

const std::string CUDARefactoringTool::skeletons[2] = {
    "skeleton_direct_kernel.cu", "skeleton_kernel.cu"};
const std::string
    CUDARefactoringTool::commandlineParams[CUDARefactoringTool::numParams] = {};
} // namespace OP2

#endif
