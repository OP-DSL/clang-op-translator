#ifndef CUDAREFACTORINGTOOL_H
#define CUDAREFACTORINGTOOL_H

#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "generators/common/GeneratorBase.hpp"
#include "generators/cuda/CudaKernelHandler.h"
#include "generators/sequential/SeqKernelHandler.h"

namespace OP2 {
/// \brief Utility of generate CUDA kernel based on ParLoop information.
///
///
class CUDARefactoringTool : public OP2KernelGeneratorBase {
  static const std::string skeletons[3];

  /// @brief Handler for CUDA specific modifications.
  ///
  CUDAKernelHandler cudaKernelHandler;

  Staging staging;

  static int getLoopType(const ParLoop &loop, Staging staging) {
    if (loop.isDirect())
      return 0;
    if (staging == OP2::OP_COlOR2)
      return 1;
    return 2;
  }

public:
  /// @brief Construct a refactoring tool to generate the CUDA kernel.
  ///
  /// @param Compilations The CompilationDatabase which contains the copmile
  /// commandlines.
  /// @param app Collected application data
  /// @param idx index of currently generated loop
  CUDARefactoringTool(const clang::tooling::CompilationDatabase &Compilations,
                      const OP2Application &app, size_t idx, Staging staging)
      : OP2KernelGeneratorBase(
            Compilations,
            {std::string(SKELETONS_DIR) +
             skeletons[getLoopType(app.getParLoops()[idx], staging)]},
            app, idx, CUDARefactoringTool::_postfix),
        cudaKernelHandler(&getReplacements(), app, idx, staging),
        staging(staging) {}

  /// @brief Adding CUDA specific Matchers and handlers.
  ///   Called from OP2KernelGeneratorBase::GenerateKernelFile()
  ///
  /// @param MatchFinder used by the RefactoringTool
  virtual void addGeneratorSpecificMatchers(
      clang::ast_matchers::MatchFinder &Finder) override {

    Finder.addMatcher(SeqKernelHandler::userFuncMatcher, &cudaKernelHandler);
    Finder.addMatcher(SeqKernelHandler::funcCallMatcher,
                      &cudaKernelHandler); // TODO update
    Finder.addMatcher(CUDAKernelHandler::cudaFuncMatcher, &cudaKernelHandler);
    Finder.addMatcher(CUDAKernelHandler::cudaFuncCallMatcher,
                      &cudaKernelHandler); // check shared
    Finder.addMatcher(CUDAKernelHandler::setReductionArraysToArgsMatcher,
                      &cudaKernelHandler);
    Finder.addMatcher(CUDAKernelHandler::setConstantArraysToArgsMatcher,
                      &cudaKernelHandler);
    Finder.addMatcher(CUDAKernelHandler::arg0hDeclMatcher,
                      &cudaKernelHandler); // check
    Finder.addMatcher(CUDAKernelHandler::mapidxDeclMatcher,
                      &cudaKernelHandler); // check + separate to 2
    Finder.addMatcher(CUDAKernelHandler::updateRedArrsOnHostMatcher,
                      &cudaKernelHandler);
    Finder.addMatcher(CUDAKernelHandler::opReductionMatcher,
                      &cudaKernelHandler);
    Finder.addMatcher(CUDAKernelHandler::declLocalRedArrMatcher,
                      &cudaKernelHandler); // check
    Finder.addMatcher(CUDAKernelHandler::initLocalRedArrMatcher,
                      &cudaKernelHandler); // check
  }

  static constexpr const char *_postfix = "kernel";
  static constexpr unsigned numParams = 4;
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

const std::string CUDARefactoringTool::skeletons[3] = {
    "cuda/skeleton_direct_kernel.cu", "cuda/skeleton_global_kernel.cu",
    "cuda/skeleton_hierarchical_kernel.cu"};
const std::string
    CUDARefactoringTool::commandlineParams[CUDARefactoringTool::numParams] = {
        std::string("-include") + OP2_INC + "op_cuda_rt_support.h",
        std::string("-include") + OP2_INC + "op_cuda_reduction_supp.h",
        std::string("-include") + OP2_INC + "op_cuda_reduction.h",
        "--cuda-device-only"};
} // namespace OP2

#endif
