#include "op_par_loop.h"

namespace OP2 {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const OP_accs_type &accs) {
  constexpr const char *OP_accs_labels[6] = {"OP_READ", "OP_WRITE", "OP_RW",
                                             "OP_INC",  "OP_MAX",   "OP_MIN"};
  return os << OP_accs_labels[accs];
}

// OPArg
DummyOPArg::DummyOPArg(const clang::VarDecl *dat, int _idx,
                       const clang::VarDecl *_map, size_t _dim,
                       std::string _type, OP_accs_type _accs)
    : op_dat(dat), idx(_idx), map(_map), dim(_dim), type(_type), accs(_accs),
      isGBL(false) {}
DummyOPArg::DummyOPArg(const clang::VarDecl *dat, size_t _dim,
                       std::string _type, OP_accs_type _accs)
    : op_dat(dat), idx(0), map(nullptr), dim(_dim), type(_type), accs(_accs),
      isGBL(true) {}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DummyOPArg &arg) {
  os << "op_arg" << (arg.isGBL ? "_gbl" : "") << ":\n\t"
     << "op_dat: " << arg.op_dat->getType().getAsString() << " "
     << arg.op_dat->getNameAsString() << "\n\t";
  // TODO give some acces to the SourceManager
  // arg.op_dat->getDefinition(arg.op_dat->getASTContext())->getLocation().print(os,
  // );
  if (!arg.isGBL) {
    if (arg.map) { // indirect argument
      os << "map: " << arg.map->getType().getAsString() << " "
         << arg.map->getNameAsString() << "\n\t";
      os << "map_idx: " << arg.idx << "\n\t";
    } else {
      os << "map: OP_ID"
         << "\n\t";
    }
  }
  return os << "dim: " << arg.dim << "\n\ttype: " << arg.type
            << "\n\taccess: " << arg.accs << "\n";
}

bool DummyOPArg::isDirect() const { return map == nullptr; }

// ParLoop functions
DummyParLoop::DummyParLoop(const clang::FunctionDecl *_function,
                           std::string _name, std::vector<OPArg> _args)
    : function(_function), name(_name), args(_args) {}

bool DummyParLoop::isDirect() const {
  return std::all_of(args.begin(), args.end(),
                     [](const OPArg &a) { return a.isDirect(); });
}

std::string DummyParLoop::getName() const { return name; }

const clang::FunctionDecl *DummyParLoop::getFunctionDecl() const {
  return function;
}

std::string DummyParLoop::getUserFuncInc(/*const clang::SourceManager& SM*/) const {
  //TODO get proper include with SourceManager instead oh hardcoded version
  return "#include \"" + name + ".h\"";
}

std::string DummyParLoop::getParLoopDef() const {
  std::string param = "void par_loop_"+name+"(const char *name, op_set set";
  llvm::raw_string_ostream os(param);
  for(size_t i = 2; i < args.size(); ++i){
    os << ", op_arg arg" << i - 2;
  }
  os << ")";
  return os.str();
}

std::vector<OPArg>::iterator DummyParLoop::arg_begin() { return args.begin(); }

std::vector<OPArg>::iterator DummyParLoop::arg_end() { return args.end(); }

} // namespace OP2
