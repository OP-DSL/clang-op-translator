#ifndef OP2GENERATORBASE_HPP
#define OP2GENERATORBASE_HPP

#include "../OP2WriteableRefactoringTool.hpp"
#include "BaseKernelHandler.h"

namespace OP2 {

/// @brief The Base class of OP2 kernel generators.
///   All subclass should implement addGeneratorSpecificMatchers function
///    and set the postfix for kernel files.
class OP2KernelGeneratorBase : public OP2WriteableRefactoringTool {
protected:
  /// @brief postfix to generate output filename
  const std::string postfix;
  /// @brief op_par_loop processed
  const ParLoop &loop;
  /// @brief Handler for common modifications
  BaseKernelHandler baseKernelHandler;

  /// @brief Generate the name of the output file.
  ///
  /// @param Entry The input file that processed
  ///
  /// @return output filename (e.g. xxx_op.cpp, xxx_kernel.cpp)
  ///   Default implementation gives: <loopname>_<postfix>.cpp
  virtual std::string getOutputFileName(const clang::FileEntry *) const {
    return loop.getName() + "_" + postfix + ".cpp";
  }

  /// @brief Adding matchers and handlers to the Finder.
  ///   Should be implemented in subclasses.
  ///
  /// @param
  virtual void
  addGeneratorSpecificMatchers(clang::ast_matchers::MatchFinder &) = 0;

  /// @brief Adding Matchers and handlers Common for all types of kernels.
  ///
  /// @param Finder
  void addBaseKernelMatchers(clang::ast_matchers::MatchFinder &Finder) {
    // Create Callbacks
    // TODO Finder to Refactoring tool?
    Finder.addMatcher(BaseKernelHandler::parLoopDeclMatcher,
                      &baseKernelHandler);
    Finder.addMatcher(BaseKernelHandler::nargsMatcher, &baseKernelHandler);
    Finder.addMatcher(BaseKernelHandler::argsArrMatcher, &baseKernelHandler);
    Finder.addMatcher(BaseKernelHandler::argsArrSetterMatcher,
                      &baseKernelHandler);
    Finder.addMatcher(BaseKernelHandler::opTimingReallocMatcher,
                      &baseKernelHandler);
    Finder.addMatcher(BaseKernelHandler::printfKernelNameMatcher,
                      &baseKernelHandler);
    Finder.addMatcher(BaseKernelHandler::opKernelsSubscriptMatcher,
                      &baseKernelHandler);
  }

public:
  OP2KernelGeneratorBase(
      const clang::tooling::CompilationDatabase &Compilations,
      const std::vector<std::string> &Sources, const ParLoop &loop,
      std::string postfix,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : OP2WriteableRefactoringTool(Compilations, Sources, PCHContainerOps),
        postfix(postfix), loop(loop),
        baseKernelHandler(&getReplacements(), loop) {}

  /// @brief Generate the kernel to <loopname>_xxkernel.cpp
  ///
  /// @return 0 on success
  int generateKernelFile() {

    // Create Callbacks
    // TODO Finder to Refactoring tool?
    clang::ast_matchers::MatchFinder Finder;
    addBaseKernelMatchers(Finder);
    // end of Base modifications
    addGeneratorSpecificMatchers(Finder);

    // run the tool
    if (int Result =
            run(clang::tooling::newFrontendActionFactory(&Finder).get())) {
      llvm::outs() << "Error " << Result << "\n";
      return Result;
    }

    // Write out replacements
    writeOutReplacements();
    return 0;
  }

  virtual ~OP2KernelGeneratorBase() = default;
};

} // namespace OP2

#endif /* ifndef OP2GENERATORBASE_HPP */
