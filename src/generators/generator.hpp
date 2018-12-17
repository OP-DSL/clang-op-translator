#ifndef GENERATOR_H
#define GENERATOR_H
#include "generators/common/LoopGenerator.hpp"
#include "generators/common/globals_header_generator.hpp"
namespace op_dsl {

class Generator {
private:
  std::unique_ptr<const GlobalsHeaderGenerator> headerGenerator;
  std::unique_ptr<LoopGenerator> loopGenerator;
  /*const AppBackendGenerator *const backendGenerator; //TODO*/
public:
  Generator(std::unique_ptr<const GlobalsHeaderGenerator> &&headerGen,
            std::unique_ptr<LoopGenerator> &&loopGen) noexcept
      : headerGenerator(std::move(headerGen)),
        loopGenerator(std::move(loopGen)) {}

  Generator(Generator &&) noexcept = default;
  Generator &operator=(Generator &&) noexcept = default;
  Generator(const Generator &) noexcept = delete;
  Generator &operator=(const Generator &) noexcept = delete;
  virtual ~Generator() = default;

  [[nodiscard]] int generate() {
    headerGenerator->generate();
    if (int err = loopGenerator->generate()) {
      return err;
    }
    /* TODO backedGeneration */
    return 0;
  }
};

} // namespace op_dsl
#endif /* GENERATOR_H */
