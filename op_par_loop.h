#ifndef __OP_PAR_LOOP_H__
#define __OP_PAR_LOOP_H__
#include "clang/AST/Decl.h"
#include <vector>

namespace OP2 {

enum OP_accs_type { OP_READ = 0, OP_WRITE, OP_RW, OP_INC, OP_MAX, OP_MIN };

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const OP_accs_type &);

class DummyOPArg {
  const clang::VarDecl *op_dat;
  int idx;
  const clang::VarDecl *map;
  size_t dim;
  std::string type;
  OP_accs_type accs;
  const bool isGBL;

public:
  DummyOPArg(const clang::VarDecl *dat, int _idx, const clang::VarDecl *_map,
             size_t _dim, std::string _type, OP_accs_type _accs);
  DummyOPArg(const clang::VarDecl *dat, size_t _dim,
             std::string _type, OP_accs_type _accs);
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const DummyOPArg&);
};
typedef DummyOPArg OPArg;

class DummyParLoop {
  const clang::FunctionDecl *function;
  const std::string name;
  std::vector<OPArg> args;

public:
  //TODO check pointers
  DummyParLoop(const clang::FunctionDecl *_function, std::string _name,
      std::vector<OPArg> _args);
};

typedef DummyParLoop ParLoop;
} // namespace OP2

#endif // end of header guard
