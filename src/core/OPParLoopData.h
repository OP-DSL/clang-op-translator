#ifndef OPPARLOOP_H
#define OPPARLOOP_H
#include "clang/AST/Decl.h"
#include <set>
#include <vector>

namespace OP2 {

/**
 * @brief Struct for olding information about global constants defined using
 * op_decl_const calls.
 *
 * These global constant might be used in the user functions as well therefore
 * the tool should be avare of these to mvoe them to the proper memory
 * environment if necessary.
 */
struct op_global_const {
  std::string type, name;
  unsigned size;
  op_global_const(std::string, std::string, unsigned);
  bool operator<(const op_global_const &) const;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const op_global_const &);
std::ostream &operator<<(std::ostream &, const op_global_const &);

/**
 * @brief Access descriptors from OP2
 */
enum OP_accs_type { OP_READ = 0, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX };

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const OP_accs_type &accs) {
  constexpr const char *OP_accs_labels[6] = {"OP_READ", "OP_WRITE", "OP_RW",
                                             "OP_INC",  "OP_MIN",   "OP_MAX"};
  return os << OP_accs_labels[accs];
}

struct UserFuncData {
  std::string functionDecl;
  std::string funcName;
  bool isInlineSpecified;
  std::string path;
  std::vector<std::string> paramNames;
  UserFuncData(std::string, std::string, bool, std::string,
               std::vector<std::string>);
  std::string getInlinedFuncDecl() const;
};

/**
 * @brief representation of an op_arg.
 */
struct OPArg {
  const int argIdx;        /**< Index of the argument in the loop*/
  const std::string opDat; /**< Variable name of the op_dat in op_arg call. */
  const int idx;           /**< Mapping index. */
  const std::string opMap; /**< Variable name of the op_map.*/
  const size_t dim;        /**< Dimension of the data. 0 if we don't know.*/
  const std::string type;  /**< Datatype held in op_dat */
  const OP_accs_type accs; /**< Access descriptor */
  // TODO move this to an enum GBL, Reduce,etc
  const bool isGBL; /**< True if the arg is a global one.*/

  OPArg(int, std::string, int, const std::string &, size_t, std::string,
        OP_accs_type);
  OPArg(int, std::string, size_t, std::string, OP_accs_type);
  bool isDirect() const;
  bool isReduction() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const OPArg &);
};

/**
 * @brief Representation of a par_loop.
 *
 * Contains all te information about a par_loop required for code generation.
 */
class ParLoop {
  inline static size_t numLoops = 0; /**< helper variable for loopId gen. */
  int loopId; /**< The unique Id of the loop used to index OP_kernels array. */
  UserFuncData function;  /**< Helds the information about the user function. */
  const std::string name; /**< Name of the user_function. This name is unique.*/
  std::vector<OPArg> args; /**< Loop arguments. */
  // TODO add enum to determine if it is an op or an ops loop

public:
  ParLoop(UserFuncData, std::string, std::vector<OPArg> _args);

  void generateID();
  bool isDirect() const;
  std::string getName() const;
  std::string getParLoopDef() const;
  size_t getLoopID() const;
  UserFuncData getUserFuncInfo() const;
};

/**
 * @brief A set of loops and constants representing the application
 *
 * This struct contains all the information necessary for the code generation
 * phase.
 */
struct OPApplication {
  std::vector<std::string>
      applicationFiles; /**< The application files containing par_loop calls. */
  std::vector<ParLoop> loops; /**< The set of par_loops of the application. */
  std::set<op_global_const> constants; /**< The set of global constants. */
  std::string applicationName;

  OPApplication() = default;
  OPApplication(const OPApplication &) = delete;

  void setName(std::string);
  std::vector<ParLoop> &getParLoops();
  const std::vector<ParLoop> &getParLoops() const;
  bool addParLoop(ParLoop);
};

} // namespace OP2

#endif /* ifndef OPPARLOOP_H */
