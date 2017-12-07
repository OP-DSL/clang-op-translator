#ifndef KERNELGENERATORSKELETON_HPP
#define KERNELGENERATORSKELETON_HPP value
#include "KernelHandlerSkeleton.hpp"
#include "core/OPParLoopData.h"
#include "generators/common/GeneratorBase.hpp"

namespace OP2 {

class KernelGeneratorSkeleton : public OP2KernelGeneratorBase {
  static const std::string skeletons[2];

  KernelHandlerSkeleton kernelHandlerSkeleton;

public:
  KernelGeneratorSkeleton(
      const clang::tooling::CompilationDatabase &Compilations,
      const OP2Application &app, size_t idx,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : OP2KernelGeneratorBase(Compilations,
                               {std::string(SKELETONS_DIR) +
                                skeletons[!app.getParLoops()[idx].isDirect()]},
                               app, idx, KernelGeneratorSkeleton::_postfix,
                               PCHContainerOps),
        kernelHandlerSkeleton(&getReplacements(), app, idx) {}

  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &Finder) {
    Finder.addMatcher(KernelHandlerSkeleton::ExampleMatcher,
                      &kernelHandlerSkeleton);
  };

  static constexpr const char *_postfix = "mypostfix";
  static constexpr unsigned numParams = 0;
  static constexpr const char *commandlineParams[numParams] = {};
};

const std::string KernelGeneratorSkeleton::skeletons[2] = {
    "direct_skeleton.cpp", "indirect_skeleton.cpp"};

} // namespace OP2

#endif /* ifndef KERNELGENERATORSKELETON_HPP */
