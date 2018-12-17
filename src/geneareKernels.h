#ifndef GENERATEKELNELS_H
#define GENERATEKELNELS_H
#include "core/OPParLoopData.h"
#include "core/clang_op_translator_core.h"
#include "core/utils.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "generators/GeneratorFactory.h"
#include <memory>

namespace op_dsl {

int generateKernelFiles(const OPOptimizations &opt, const OPTargets &opTarget,
                        clang::tooling::CommonOptionsParser &optionsParser,
                        const OPApplication &app, const DSL &dsl) {
  if (opTarget == none)
    return 0;
  //TODO
  /*std::vector<std::string> args = op_dsl::getCommandlineArgs(optionsParser);
  args.insert(args.begin(), std::string("-I") + OP2_INC);
  clang::tooling::FixedCompilationDatabase Compilations(".", args);*/
  
  std::unique_ptr<GeneratorFactory> factory;

  if(dsl == OP2) {
     factory = std::make_unique<op2::OP2GeneratorFactory>(app, opt, optionsParser); 
  } else {
     factory = std::make_unique<ops::OPSGeneratorFactory>(app, opt, optionsParser); 
  }
  if (opTarget == seq || opTarget == all) { //generate?...
    Generator generator = factory->createSequentialGenerator();
    if(int err = generator.generate()) {
      return err;
    }
  }
  /*
  if (opTarget == openmp || opTarget == all) {
    OpenMPGenerator generator(application, commandLineArgs, Compilations,
                              optimizationFlags);
    generator.generateKernelFiles();
  }
  if (opTarget == vec || opTarget == all) {
    VectorizedGenerator generator(application, commandLineArgs, Compilations,
                                  optimizationFlags, "skeleton_veckernels.cpp");
    generator.generateKernelFiles();
  }
  if (opTarget == cuda || opTarget == all) {
    CUDAGenerator generator(application, commandLineArgs, Compilations,
                            optimizationFlags, "skeleton_kernels.cu");
    generator.generateKernelFiles();
  }
  */
  return 0;
}

} // namespace op_dsl
#endif /* ifndef GENERATEKERNELS_H */
