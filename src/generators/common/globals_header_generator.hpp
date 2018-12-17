#ifndef GLOBALS_HEADER_GENERATOR_H
#define GLOBALS_HEADER_GENERATOR_H
#include "core/OPParLoopData.h"
#include <algorithm>
#include <fstream>

namespace op_dsl {

/**
 * @brief Small class for generating a header containing declarations for global
 * constants to <applicationname>_globals.h.
 *
 */
class GlobalsHeaderGenerator {
private:
  const OPApplication &app;
  DSL dsl;

public:
  GlobalsHeaderGenerator(const OPApplication &app, DSL dsl) noexcept
      : app(app), dsl(dsl) {}
  void generate() const {
    std::string filename = app.applicationName + "_globals";
    std::ofstream of(filename + ".h");
    std::transform(filename.begin(), filename.end(), filename.begin(), toupper);
    of << "#ifndef " << filename << "_H\n";
    of << "#define " << filename << "_H\n";
    if (dsl == OP2) {
      of << "#include \"op_lib_cpp.h\"\n";
    } else {
      of << "#include \"ops_lib_cpp.h\"\n";
    }
    for (const auto &c : app.constants) {
      of << "extern " << c << ";\n";
    }
    of << "#endif // " << filename << "_H\n";
    of.close();
  }
  virtual ~GlobalsHeaderGenerator() = default;
};

} // namespace op_dsl
#endif /* GLOBALS_HEADER_GENERATOR_H */
