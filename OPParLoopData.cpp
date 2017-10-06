#include "OPParLoopData.h"

namespace OP2 {

DummyOPArgv2::DummyOPArgv2(const clang::VarDecl *dat, int _idx,
                           const clang::VarDecl *_map, size_t _dim,
                           std::string _type, OP_accs_type _accs)
    : opDat(dat->getNameAsString()), idx(_idx), opMap(!_map), dim(_dim),
      type(_type), accs(_accs), isGBL(false) {}

DummyOPArgv2::DummyOPArgv2(const clang::VarDecl *dat, size_t _dim,
                           std::string _type, OP_accs_type _accs)
    : opDat(dat->getNameAsString()), idx(0), opMap(false), dim(_dim),
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
    ss << args[i].getArgCall(i, "n") << ",\n";
  }
  ss << args[args.size() - 1].getArgCall(args.size() - 1, "n") << "\n";
  ss << ");";
  return ss.str();
}

} // namespace OP2
