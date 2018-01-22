#ifndef MASTERKERNELGENERATOR_HPP
#define MASTERKERNELGENERATOR_HPP
#include "core/OP2WriteableRefactoringTool.hpp"
#include "generators/common/handler.hpp"
#include "generators/openmp/OMPRefactoringTool.h"
#include "generators/sequential/SeqRefactoringTool.h"
#include "generators/vectorization/VecRefactoringTool.h"
#include <algorithm>
#include <fstream>

namespace OP2 {
namespace {
const DeclarationMatcher globVarMatcher =
    varDecl(hasName("global_var")).bind("global_var");
}

namespace matchers = clang::ast_matchers;
class MasterKernelHandler : public matchers::MatchFinder::MatchCallback {
  const std::vector<std::string> &kernels;
  const std::set<op_global_const> &constants;

  std::map<std::string, clang::tooling::Replacements> *Replace;

public:
  MasterKernelHandler(
      std::vector<std::string> &kernels,
      const std::set<op_global_const> &constants,
      std::map<std::string, clang::tooling::Replacements> *Replace)
      : kernels(kernels), constants(constants), Replace(Replace) {}

  std::string generateFile() {
    std::string repl;
    llvm::raw_string_ostream os(repl);
    for (const op_global_const &c : constants) {
      os << "extern " << c << ";\n";
    }
    os << "// user kernel files\n";
    for (const std::string &kernel : kernels) {
      os << "#include \"" << kernel << "\"\n";
    }
    return os.str();
  }

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override {
    HANDLER(clang::VarDecl, 11, "global_var",
            MasterKernelHandler::generateFile);
  }
};

template <typename KernelGeneratorType, typename Handler = MasterKernelHandler>
class MasterkernelGenerator : public OP2WriteableRefactoringTool {
protected:
  const OP2Application &application;
  std::vector<std::string> commandLineArgs;
  std::vector<std::string> generatedFiles;

public:
  MasterkernelGenerator(const OP2Application &app,
                        std::vector<std::string> &args,
                        clang::tooling::FixedCompilationDatabase &compilations,
                        std::string skeleton_name = "skeleton_kernels.cpp")
      : OP2WriteableRefactoringTool(
            compilations, {std::string(SKELETONS_DIR) + skeleton_name}),
        application(app), commandLineArgs(args) {}

  /// @brief Generates kernelfiles for all parLoop and then the master kernel
  ///  file
  void generateKernelFiles() {
    for (size_t i = 0; i < KernelGeneratorType::numParams; ++i) {
      commandLineArgs.push_back(KernelGeneratorType::commandlineParams[i]);
    }
    commandLineArgs.push_back(std::string("-include") + OP2_INC +
                              "op_lib_cpp.h");
    // Currently this only needed when we modify the user kernel.
    commandLineArgs.push_back(std::string("-xc++"));

    std::ofstream os("/tmp/" + application.applicationName + "_global.h");
    for (const op_global_const &c : application.constants) {
      os << c << ";\n";
    }
    os.close();
    commandLineArgs.push_back("-include/tmp/" + application.applicationName +
                              "_global.h");

    clang::tooling::FixedCompilationDatabase F(".", commandLineArgs);
    for (size_t i = 0; i < application.getParLoops().size(); ++i) {

      KernelGeneratorType tool(F, application, i);
      if (tool.generateKernelFile()) {
        llvm::outs() << "Error during processing ";
      }
      llvm::outs() << application.getParLoops()[i].getName() << "\n";
      generatedFiles.push_back(tool.getOutputFileName());
    }

    clang::ast_matchers::MatchFinder Finder;
    Handler handler(generatedFiles, application.constants, &getReplacements());
    Finder.addMatcher(globVarMatcher, &handler);
    if (int Result =
            run(clang::tooling::newFrontendActionFactory(&Finder).get())) {
      llvm::outs() << "Error " << Result << "\n";
      return;
    }
    writeOutReplacements();
  }

  std::string getOutputFileName(const clang::FileEntry *) const {
    return application.applicationName + "_" + KernelGeneratorType::_postfix +
           "s.cpp";
  }
};

typedef MasterkernelGenerator<SeqRefactoringTool> SeqGenerator;
typedef MasterkernelGenerator<OMPRefactoringTool> OpenMPGenerator;
typedef MasterkernelGenerator<VecRefactoringTool> VectorizedGenerator;
typedef MasterkernelGenerator<SeqRefactoringTool> CUDAGenerator;
} // namespace OP2

#endif /* ifndef MASTERKERNELGENERATOR_HPP */
