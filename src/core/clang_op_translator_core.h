#ifndef CLANG_OP_TRANSLATOR_CORE_H
#define CLANG_OP_TRANSLATOR_CORE_H
/*
 * This header file declares enums and types for op2_clang.
 * */

namespace op_dsl {

/// @brief Enum for representing which we want to use (OP2 or OPS)
enum DSL { OP2 = 0, OPS };

/// @brief Enum type for optarget commandline option.
enum OPTargets { all = 0, none, seq, openmp, vec, cuda };

/// @brief Enum type to set coloring type for cuda.
enum Staging { OP_STAGE_ALL = 0, OP_COlOR2 };

/// @brief Struct for bundle optimization settings.
///
struct OPOptimizations {
  Staging staging;
  bool SOA;
};

} // namespace op_dsl

#endif /* ifndef CLANG_OP_TRANSLATOR_CORE_H */
