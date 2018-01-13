#ifndef OP2REFACTORINGTOOL_HPP
#define OP2REFACTORINGTOOL_HPP

// OP2 includes
#include "OPDataRegister.hpp"
#include "ParLoopHandler.hpp"
#include "core/OP2WriteableRefactoringTool.hpp"

#include "generators/MasterkernelGenerator.hpp"

namespace OP2 {

enum OP2Targets { all = 0, none, seq, openmp, vec };

class OP2RefactoringTool : public OP2WriteableRefactoringTool {
protected:
  OP2Targets opTarget;
  std::vector<std::string> &commandLineArgs;
  clang::tooling::FixedCompilationDatabase &Compilations;
  // We can collect all data about kernels
  OP2Application application;

public:
  OP2RefactoringTool(
      std::vector<std::string> &commandLineArgs,
      clang::tooling::FixedCompilationDatabase &compilations,
      clang::tooling::CommonOptionsParser &optionsParser, OP2Targets opTarget,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : OP2WriteableRefactoringTool(
            compilations, optionsParser.getSourcePathList(), PCHContainerOps),
        opTarget(opTarget), commandLineArgs(commandLineArgs),
        Compilations(compilations), application() {
    std::string applicationName = optionsParser.getSourcePathList()[0];
    size_t basename_start = applicationName.rfind("/"),
           basename_end = applicationName.rfind(".");
    if (basename_start == std::string::npos) {
      basename_start = 0;
    } else {
      basename_start++;
    }
    if (basename_end == std::string::npos || basename_end < basename_start)
      llvm::errs() << "Invalid applicationName: " << applicationName << "\n";
    applicationName =
        applicationName.substr(basename_start, basename_end - basename_start);
    application.setName(applicationName);
  }

  /// @brief Generates kernelfiles for all parLoop
  void generateKernelFiles() {
    if (opTarget == none)
      return;
    if (opTarget == seq || opTarget == all) {
      SeqGenerator generator(application, commandLineArgs, Compilations);
      generator.generateKernelFiles();
    }
    if (opTarget == openmp || opTarget == all) {
      OpenMPGenerator generator(application, commandLineArgs, Compilations);
      generator.generateKernelFiles();
    }
    if (opTarget == vec || opTarget == all) {
      VectorizedGenerator generator(application, commandLineArgs, Compilations,
                                    "skeleton_veckernels.cpp");
      generator.generateKernelFiles();
    }
  }

  /// @brief Setting the finders for the refactoring tool then runs the tool
  ///   to collect data about the op calls in the application and generate the
  ///   xxx_op files.
  ///
  /// @return 0 on success
  int generateOPFiles() {
    clang::ast_matchers::MatchFinder Finder;
    // creating and setting callback to handle op_par_loops
    ParLoopHandler parLoopHandlerCallback(&getReplacements(), application);
    Finder.addMatcher(
        callExpr(callee(functionDecl(hasName("op_par_loop")))).bind("par_loop"),
        &parLoopHandlerCallback);

    DataRegister drCallback(application);
    drCallback.addToFinder(Finder);

    return run(clang::tooling::newFrontendActionFactory(&Finder).get());
  }

  /// @brief Generate output filename from the filename of the processed file.
  ///
  /// @param Entry The input file that processed.
  ///
  /// @return output filename (xxx_op.cpp)
  virtual std::string getOutputFileName(const clang::FileEntry *Entry) const {
    llvm::outs() << "Rewrite buffer for file: " << Entry->getName() << "\n";

    std::string filename = Entry->getName().str();
    size_t basename_start = filename.rfind("/") + 1,
           basename_end = filename.rfind(".");
    if (basename_start == std::string::npos)
      basename_start = 0;
    if (basename_end == std::string::npos || basename_end < basename_start)
      llvm::errs() << "Invalid filename: " << Entry->getName() << "\n";
    filename = filename.substr(basename_start, basename_end - basename_start) +
               "_op.cpp";
    return filename;
  }
};

} // namespace OP2

#endif /* ifndef OP2REFACTORINGTOOL_HPP */
