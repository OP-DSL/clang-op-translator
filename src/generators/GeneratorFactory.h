#ifndef GENERATORFACTORY_H
#define GENERATORFACTORY_H
#include "generators/generator.hpp"
#include "clang/Tooling/CommonOptionsParser.h"
#include <memory>

namespace op_dsl {
class GeneratorFactory {
protected:
  const OPApplication &app;
  const OPOptimizations &opt;
  clang::tooling::CommonOptionsParser &optionsParser;

public:
  GeneratorFactory(const OPApplication &_app, const OPOptimizations &_opt,
                   clang::tooling::CommonOptionsParser &optionsParser)
      : app(_app), opt(_opt), optionsParser(optionsParser) {}
  virtual ~GeneratorFactory() = default;
  virtual op_dsl::Generator createSequentialGenerator() = 0;
  // virtual op_dsl::Generator createOpenMPGenerator() = 0;
};

namespace op2 {
class OP2GeneratorFactory : public GeneratorFactory {
public:
  OP2GeneratorFactory(const OPApplication &_app, const OPOptimizations &_opt,
                      clang::tooling::CommonOptionsParser &optionsParser)
      : GeneratorFactory(_app, _opt, optionsParser) {}
  virtual ~OP2GeneratorFactory() = default;

  op_dsl::Generator createSequentialGenerator() override {
    return Generator(
        std::make_unique<const GlobalsHeaderGenerator>(app, DSL::OP2),
        std::make_unique<LoopGenerator>(app, opt, DSL::OP2, optionsParser));
  }
  // op_dsl::Generator createOpenMPGenerator() override{}
};
} // namespace op2

namespace ops {
class OPSGeneratorFactory : public GeneratorFactory {
public:
  OPSGeneratorFactory(const OPApplication &_app, const OPOptimizations &_opt,
                      clang::tooling::CommonOptionsParser &optionsParser)
      : GeneratorFactory(_app, _opt, optionsParser) {}
  virtual ~OPSGeneratorFactory() = default;

  op_dsl::Generator createSequentialGenerator() override {
    return Generator(
        std::make_unique<const GlobalsHeaderGenerator>(app, DSL::OPS),
        std::make_unique<LoopGenerator>(app, opt, DSL::OPS, optionsParser));
  }
  // op_dsl::Generator createOpenMPGenerator() override{}
};
} // namespace ops

} // namespace op_dsl

#endif /* GENERATORFACTORY_H */

