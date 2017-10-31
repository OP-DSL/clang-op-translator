#include "OPParLoopData.h"

namespace OP2 {
//___________________________________OP_SET___________________________________

op_set::op_set(int s, std::string n) : size(s), name(n) {}
bool op_set::operator==(const op_set &_set) const {
  return size == _set.size && name == _set.name;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const op_set &s) {
  return os << s.name << "{" << s.size << "}";
}

//___________________________________OP_MAP___________________________________
op_map::op_map(const op_set &from, const op_set &to, unsigned d, std::string s)
    : from(from), to(to), dim(d), name(s) {}
bool op_map::operator==(const op_map &m) const {
  return from == m.from && to == m.to && dim == m.dim && name == m.name;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const op_map &m) {
  return os << "map:" << m.name << "::" << m.from << "-->" << m.to
            << " dim: " << m.dim;
}

//___________________________________OP_ARG___________________________________
DummyOPArgv2::DummyOPArgv2(const clang::VarDecl *dat, int _idx,
                           const clang::VarDecl *_map, size_t _dim,
                           std::string _type, OP_accs_type _accs)
    : opDat(dat->getNameAsString()), idx(_idx), opMap(!_map), dim(_dim),
      type(_type), accs(_accs), isGBL(false) {}

DummyOPArgv2::DummyOPArgv2(const clang::VarDecl *dat, size_t _dim,
                           std::string _type, OP_accs_type _accs)
    : opDat(dat->getNameAsString()), idx(0), opMap(true), dim(_dim),
      type(_type), accs(_accs), isGBL(true) {}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DummyOPArgv2 &arg) {
  os << "op_arg" << (arg.isGBL ? "_gbl" : "") << ":\n\t"
     << "op_dat: " << arg.opDat << "\n\t";
  if (!arg.isGBL) {
    if (!arg.opMap) { // indirect argument
      os << "map_idx: " << arg.idx << "\n\t";
    } else {
      os << "map: OP_ID\n\t";
    }
  }
  return os << "dim: " << arg.dim << "\n\ttype: " << arg.type
            << "\n\taccess: " << arg.accs << "\n";
}
bool DummyOPArgv2::isDirect() const { return opMap; }

std::string DummyOPArgv2::getArgCall(int argIdx, std::string mapStr) const {
  return "&((" + type + "*)arg" + std::to_string(argIdx) + ".data)[" +
         std::to_string(dim) + "*" + mapStr + "]";
}

bool DummyOPArgv2::isReduction() const {
  return isGBL && (accs == OP_INC || accs == OP_MAX || accs == OP_MIN);
}

// ParLoop functions

size_t DummyParLoop::numLoops = 0;

DummyParLoop::DummyParLoop(const clang::FunctionDecl *_function,
                           std::string _name, std::vector<OPArg> _args)
    : loopId(numLoops++), function(_function->getNameAsString()), name(_name),
      args(_args) {}

bool DummyParLoop::isDirect() const {
  return std::all_of(args.begin(), args.end(),
                     [](const OPArg &a) { return a.isDirect(); });
}

std::string DummyParLoop::getName() const { return name; }

size_t DummyParLoop::getLoopID() const { return loopId; }

std::string DummyParLoop::getUserFuncInc() const {
  // TODO get proper include
  return "#include \"" + name + ".h\"";
}

std::string DummyParLoop::getParLoopDef() const {
  std::string param = "void par_loop_" + name + "(const char *name, op_set set";
  llvm::raw_string_ostream os(param);
  for (size_t i = 2; i < args.size(); ++i) {
    os << ", op_arg arg" << i - 2;
  }
  os << ")";
  return os.str();
}

size_t DummyParLoop::getNumArgs() const { return args.size(); }

const OPArg &DummyParLoop::getArg(size_t ind) const { return args[ind]; }

unsigned DummyParLoop::getKernelType() const {
  if (isDirect()) {
    return 0;
  }
  // TODO other kernel types maybe an enum;
  return 0;
}

std::string DummyParLoop::getFuncCall() const {
  std::string funcCall = "";
  llvm::raw_string_ostream ss(funcCall);
  ss << name << "("; // TODO fix repr to store correct function data.
  for (size_t i = 0; i < args.size() - 1; ++i) {
    ss << args[i].getArgCall(
              i, args[i].isDirect() ? "n" : ("map" + std::to_string(i) + "idx"))
       << ",\n";
  }
  ss << args[args.size() - 1].getArgCall(
            args.size() - 1,
            args[args.size() - 1].isDirect()
                ? "n"
                : ("map" + std::to_string(args.size() - 1) + "idx"))
     << "\n";
  ss << ");";
  return ss.str();
}

std::string DummyParLoop::getMPIReduceCall() const {
  std::string reduceCall = "";
  llvm::raw_string_ostream ss(reduceCall);
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].isReduction()) {
      ss << "op_mpi_reduce(&arg" << i << ", (" << args[i].type << " *)arg" << i
         << ".data);\n";
    }
  }
  return ss.str();
}

std::string DummyParLoop::getTransferData() const {
  std::string transfer = "";
  llvm::raw_string_ostream ss(transfer);
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].isGBL) {
      continue;
    }
    ss << "OP_kernels[" << loopId << "].transfer += (float)set->size * arg" << i
       << ".size";
    if (args[i].accs != OP_READ) {
      ss << " * 2.0f";
    }
    ss << ";\n";
  }
  return ss.str();
}

std::string DummyParLoop::getMapVarDecls() const {
  std::string mapDecls = "";
  llvm::raw_string_ostream ss(mapDecls);
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].isDirect() || args[i].isGBL) {
      continue;
    }
    ss << "int map" << i << "idx = arg" << i << ".map_data[n * arg" << i
       << ".map->dim + " << args[i].idx << "];\n";
  }
  return ss.str();
}

} // namespace OP2
