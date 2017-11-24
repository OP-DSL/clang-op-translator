#ifndef KERNELGENERATORSKELETON_HPP
#define KERNELGENERATORSKELETON_HPP value
#include "../OPParLoopData.h"
#include "GeneratorBase.hpp"
#include "KernelHandlerSkeleton.hpp"

namespace OP2 {

class KernelGeneratorSkeleton : public OP2KernelGeneratorBase {
  static const std::string skeletons[2];

  KernelHandlerSkeleton kernelHandlerSkeleton;

public:
  KernelGeneratorSkeleton(
      const clang::tooling::CompilationDatabase &Compilations,
      const ParLoop &loop,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : OP2KernelGeneratorBase(
            Compilations,
            {std::string(SKELETONS_DIR) + skeletons[!loop.isDirect()]}, loop,
            KernelGeneratorSkeleton::_postfix, PCHContainerOps),
        kernelHandlerSkeleton(&getReplacements(), loop) {}

  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &Finder) {
    Finder.addMatcher(KernelHandlerSkeleton::ExampleMatcher,
                      &kernelHandlerSkeleton);
  };

  static constexpr const char *_postfix = "mypostfix";
};

static const std::string skeletons[2] = {"direct_skeleton.cpp",
                                         "indirect_skeleton.cpp"};

} // namespace OP2

#endif /* ifndef KERNELGENERATORSKELETON_HPP */
