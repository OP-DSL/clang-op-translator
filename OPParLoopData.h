#ifndef OPPARLOOP_H
#define OPPARLOOP_H
#include "llvm/Support/raw_ostream.h"

namespace OP2 {

enum OP_accs_type { OP_READ = 0, OP_WRITE, OP_RW, OP_INC, OP_MAX, OP_MIN };

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const OP_accs_type &accs) {
  constexpr const char *OP_accs_labels[6] = {"OP_READ", "OP_WRITE", "OP_RW",
                                             "OP_INC",  "OP_MAX",   "OP_MIN"};
  return os << OP_accs_labels[accs];
}

struct DummyOPArgv2 {
  const clang::VarDecl *op_dat;
  int idx;
  const clang::VarDecl *map;
  size_t dim;
  std::string type;
  OP_accs_type accs;
  const bool isGBL;

  DummyOPArg(const clang::VarDecl *dat, int _idx, const clang::VarDecl *_map,
             size_t _dim, std::string _type, OP_accs_type _accs);
  DummyOPArg(const clang::VarDecl *dat, size_t _dim, std::string _type,
             OP_accs_type _accs);
  bool isDirect() const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const DummyOPArg &);
};
typedef DummyOPArgv2 OPArg;

} // namespace OP2

#endif /* ifndef OPPARLOOP_H */
