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
                               {std::string(SKELETONS_DIR) +
                                skeletons[!app.getParLoops()[idx].isDirect()]},
                               app, idx, CUDARefactoringTool::_postfix),
        cudaKernelHandler(&getReplacements(), app, idx) {}

  /// @brief Adding CUDA specific Matchers and handlers.
  ///   Called from OP2KernelGeneratorBase::GenerateKernelFile()
  ///
  /// @param MatchFinder used by the RefactoringTool
  virtual void addGeneratorSpecificMatchers(
      clang::ast_matchers::MatchFinder &Finder) override {

    Finder.addMatcher(SeqKernelHandler::userFuncMatcher,
                      &cudaKernelHandler); // TODO check
    Finder.addMatcher(SeqKernelHandler::funcCallMatcher,
                      &cudaKernelHandler); // TODO update
    Finder.addMatcher(CUDAKernelHandler::cudaFuncMatcher,
                      &cudaKernelHandler); // TODO update
    Finder.addMatcher(CUDAKernelHandler::cudaFuncCallMatcher,
                      &cudaKernelHandler); // TODO update
    Finder.addMatcher(CUDAKernelHandler::setReductionArraysToArgsMatcher,
                      &cudaKernelHandler); // check
    Finder.addMatcher(CUDAKernelHandler::setConstantArraysToArgsMatcher,
                      &cudaKernelHandler); // check
    Finder.addMatcher(CUDAKernelHandler::arg0hDeclMatcher, &cudaKernelHandler); // check
    Finder.addMatcher(CUDAKernelHandler::mapidxDeclMatcher, &cudaKernelHandler); // check
    Finder.addMatcher(CUDAKernelHandler::updateRedArrsOnHostMatcher,
                      &cudaKernelHandler); // check
    Finder.addMatcher(CUDAKernelHandler::opReductionMatcher,
                      &cudaKernelHandler); // check
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

const std::string CUDARefactoringTool::skeletons[2] = {
    "cuda/skeleton_direct_kernel.cu", "cuda/skeleton_global_kernel.cu"};
const std::string
    CUDARefactoringTool::commandlineParams[CUDARefactoringTool::numParams] = {
        std::string("-include") + OP2_INC + "op_cuda_rt_support.h",
        std::string("-include") + OP2_INC + "op_cuda_reduction_supp.h",
        std::string("-include") + OP2_INC + "op_cuda_reduction.h",
        "--cuda-device-only"};
} // namespace OP2

#endif
