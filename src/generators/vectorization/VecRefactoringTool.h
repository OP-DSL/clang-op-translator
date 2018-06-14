#ifndef VECREFACTORINGTOOL_H
#define VECREFACTORINGTOOL_H value
#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "generators/common/GeneratorBase.hpp"
#include "generators/sequential/SeqKernelHandler.h"
#include "generators/vectorization/VecKernelHandler.h"

namespace OP2 {

class VecRefactoringTool : public OP2KernelGeneratorBase {
  static const std::string skeletons[2];

  VecKernelHandler vecKernelHandler;
  SeqKernelHandler seqKernelHandler;

public:
  VecRefactoringTool(const clang::tooling::CompilationDatabase &Compilations,
                     const OP2Application &app, size_t idx, OP2Optimizations op)
      : OP2KernelGeneratorBase(Compilations,
                               {std::string(SKELETONS_DIR) +
                                skeletons[!app.getParLoops()[idx].isDirect()]},
                               app, idx, VecRefactoringTool::_postfix, op),
        vecKernelHandler(&getReplacements(), Compilations, app, idx),
        seqKernelHandler(&getReplacements(), app, idx) {}

  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &Finder) {
    Finder.addMatcher(SeqKernelHandler::mapIdxDeclMatcher, &seqKernelHandler);
    Finder.addMatcher(SeqKernelHandler::userFuncMatcher, &vecKernelHandler);
    Finder.addMatcher(SeqKernelHandler::funcCallMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::vecFuncCallMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::vecUserFuncMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::alignedPtrMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::localRedVarMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::localidx0Matcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::localidx1Matcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::localIndirectVarMatcher,
                      &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::redForMatcher, &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::localIncWriteBackMatcher,
                      &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::localIndDatInitMatcher,
                      &vecKernelHandler);
    Finder.addMatcher(VecKernelHandler::localIndRedWriteBackMatcher,
                      &vecKernelHandler);
  };

  static constexpr const char *_postfix = "veckernel";
  static constexpr const char *fileExtension = ".cpp";
  static constexpr unsigned numParams = 3;
  static const std::string commandlineParams[numParams];
};

const std::string
    VecRefactoringTool::commandlineParams[VecRefactoringTool::numParams] = {
        "-DSIMD_VEC=4", "-DVECTORIZE", "-xc++"};
const std::string VecRefactoringTool::skeletons[2] = {
    "skeleton_direct_veckernel.cpp", "skeleton_veckernel.cpp"};

} // namespace OP2

#endif /* ifndef VECREFACTORINGTOOL_H */
