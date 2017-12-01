#ifndef VECREFACTORINGTOOL_H
#define VECREFACTORINGTOOL_H value
#include "../OPParLoopData.h"
#include "GeneratorBase.hpp"
#include "SeqKernelHandler.h"
#include "VecKernelHandler.h"

namespace OP2 {

class VecRefactoringTool : public OP2KernelGeneratorBase {
  static const std::string skeletons[2];

  VecKernelHandler vecKernelHandler;

public:
  VecRefactoringTool(
      const clang::tooling::CompilationDatabase &Compilations,
      const ParLoop &loop,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : OP2KernelGeneratorBase(
            Compilations,
            {std::string(SKELETONS_DIR) + skeletons[!loop.isDirect()]}, loop,
            VecRefactoringTool::_postfix, PCHContainerOps),
        vecKernelHandler(&getReplacements(), Compilations, loop) {}

  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &Finder) {
    Finder.addMatcher(SeqKernelHandler::userFuncMatcher, &vecKernelHandler);
    Finder.addMatcher(SeqKernelHandler::funcCallMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::vecFuncCallMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::vecUserFuncMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::alignedPtrMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::localRedVarMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::redForMatcher, &vecKernelHandler);
  };

  static constexpr const char *_postfix = "veckernel";
  static constexpr unsigned numParams = 2;
  static const std::string commandlineParams[numParams];
};

const std::string
    VecRefactoringTool::commandlineParams[VecRefactoringTool::numParams] = {
        "-DSIMD_VEC=4", "-DVECTORIZE"};
const std::string VecRefactoringTool::skeletons[2] = {
    "skeleton_direct_veckernel.cpp", "skeleton_veckernel.cpp"};

} // namespace OP2

#endif /* ifndef VECREFACTORINGTOOL_H */
