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
 * @brief Access descriptors from OP2 and OPS
 */
enum OP_accs_type { OP_READ = 0, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX };

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const OP_accs_type &accs) {
  constexpr const char *OP_accs_labels[6] = {"OP_READ", "OP_WRITE", "OP_RW",
                                             "OP_INC",  "OP_MIN",   "OP_MAX"};
  return os << OP_accs_labels[accs];
}

/**
 * @brief Enum to determine the type of a par_loop
 */
enum OPLoopKind { OP2 = 0, OPS = 1 };

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &o,
                                     const OPLoopKind &kind) {
  constexpr const char *OPLoopKind_labels[2] = {"OP2", "OPS"};
  return o << OPLoopKind_labels[kind];
}

/**
 * @brief representation of the elementary function passed to a par_loop call
 *
 * Collects all data about the user given elementary operation of the par_loop
 */
struct UserFuncData {
  std::string functionDecl; /**< The source of the elementary operation */
  std::string funcName;     /**< The function name from the funcDecl */
  bool isInlineSpecified;   /**< true if the parsed funcDecl is has inline */
  std::string path; /**< Path to the file where the function is defined */
  std::vector<std::string> paramNames; /**< Names of the function arguments */
  UserFuncData(std::string, std::string, bool, std::string,
               std::vector<std::string>);
  std::string getInlinedFuncDecl() const;
};

/**
 * @brief representation of an op_arg.
 */
struct OPArg {
  /**
   * @brief Type of the argument i.e. global or data
   */
  enum OPArgKind { OP_GBL = 0, OP_DAT, OP_REDUCE, OP_IDX };

  const size_t argIdx;     /**< Index of the argument in the loop */
  const std::string opDat; /**< Variable name of the op_dat in op_arg call */
  const size_t dim;        /**< Dimension of the data, 0 if unknown.*/
  const std::string type;  /**< Datatype held in op_dat */
  const OP_accs_type accs; /**< Access descriptor */
  const OPArgKind kind;    /**< Global arg or it come from an op_dat */
  const int mapIdx;        /**< Mapping index (-1 if direct) */
  const std::string opMap; /**< Variable name of the op_map. The value ""
                              iff the arg is direct */
  const bool optional;     /**< True if the arg is optional i.e. declared with
                              ops_arg_dat_opt or op_opt_arg_dat */
  OPArg(size_t idx, std::string datName, size_t datDim, std::string type,
        OP_accs_type accs, OPArgKind kind, int mapIdx = -1,
        std::string mapName = "", bool opt = false);
  OPArg(size_t idx);
  bool isDirect() const;
  bool isReduction() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const OPArg &);
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &o,
                                     const OPArg::OPArgKind &kind) {
  constexpr const char *OPArgKind_labels[4] = {"OP_GBL", "OP_DAT", "OP_REDUCE",
                                               "OP_IDX"};
  return o << OPArgKind_labels[kind];
}

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
  OPLoopKind loopKind; /**< Store the type of the loop (wheter it is an op or an
                          ops_par_loop) */

public:
  ParLoop(UserFuncData, std::string, std::vector<OPArg>, OPLoopKind kind = OP2);

  void generateID();
  bool isDirect() const;
  std::string getName() const;
  std::string getParLoopDef() const;
  size_t getLoopID() const;
  UserFuncData getUserFuncInfo() const;
  void prettyPrint(llvm::raw_ostream &) const;
  OPLoopKind getKind() const;
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
  std::set<op_global_const>
      constants; /**< The set of global constants defined by op_decl_const. */
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
