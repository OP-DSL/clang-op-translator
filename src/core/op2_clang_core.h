#ifndef OP2_CLANG_CORE_H
#define OP2_CLANG_CORE_H
/*
 * This header file declares enums and types for op2_clang.
 * */

namespace OP2 {
/// @brief Enum type for optarget commandline option.
enum OP2Targets { all = 0, none, seq, openmp, vec, cuda };
/// @brief Enum type to set coloring type for cuda.
enum Staging { OP_STAGE_ALL = 0, OP_COlOR2 };

/// @brief Struct for bundle optimization settings.
///
struct OP2Optimizations {
  Staging staging;
  bool SOA;
};

} // namespace OP2

#endif /* ifndef OP2_CLANG_CORE_H */
