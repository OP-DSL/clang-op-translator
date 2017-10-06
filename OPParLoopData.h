#ifndef OPPARLOOP_H
#define OPPARLOOP_H
#include "clang/AST/Decl.h"
#include <vector>

namespace OP2 {

enum OP_accs_type { OP_READ = 0, OP_WRITE, OP_RW, OP_INC, OP_MAX, OP_MIN };

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const OP_accs_type &accs) {
  constexpr const char *OP_accs_labels[6] = {"OP_READ", "OP_WRITE", "OP_RW",
                                             "OP_INC",  "OP_MAX",   "OP_MIN"};
  return os << OP_accs_labels[accs];
}

struct DummyOPArgv2 {
  std::string opDat;
  int idx;
  bool opMap;
  size_t dim;
  std::string type;
  OP_accs_type accs;
  const bool isGBL;

  DummyOPArgv2(const clang::VarDecl *, int, const clang::VarDecl *, size_t,
               std::string, OP_accs_type);
  DummyOPArgv2(const clang::VarDecl *, size_t, std::string, OP_accs_type);
  bool isDirect() const;
  std::string getArgCall(int, std::string) const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const DummyOPArgv2 &);
};

typedef DummyOPArgv2 OPArg;

class DummyParLoop {
  static size_t numLoops;
  int loopId;
  std::string function;
  const std::string name;
  std::vector<OPArg> args;

public:
  DummyParLoop(const clang::FunctionDecl *_function, std::string _name,
               std::vector<OPArg> _args);

  bool isDirect() const;
  std::string getName() const;
  std::string getFuncCall() const;
  std::string getUserFuncInc() const;
  std::string getParLoopDef() const;
  size_t getNumArgs() const;
  size_t getLoopID() const;
  unsigned getKernelType() const;
};

typedef DummyParLoop ParLoop;

} // namespace OP2

#endif /* ifndef OPPARLOOP_H */
