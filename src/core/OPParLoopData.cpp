#include "OPParLoopData.h"
#include "core/utils.h"
#include <fstream>

namespace OP2 {
//__________________________________OP_CONST__________________________________
op_global_const::op_global_const(std::string T, std::string name, unsigned S)
    : type(T), name(name), size(S) {}

bool op_global_const::operator<(const op_global_const &c) const {
  return name < c.name;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const op_global_const &c) {
  os << c.type << " " << c.name;
  if (c.size != 1)
    os << "[" << c.size << "]";
  return os;
}
std::ostream &operator<<(std::ostream &os, const op_global_const &c) {
  os << c.type << " " << c.name;
  if (c.size != 1)
    os << "[" << c.size << "]";
  return os;
}

//___________________________________OP_MAP___________________________________
op_map::op_map(const op_set &from, const op_set &to, unsigned d, std::string s)
    : from(from), to(to), dim(d), name(s) {}
bool op_map::operator==(const op_map &m) const {
  return from == m.from && to == m.to && dim == m.dim && name == m.name;
}
bool op_map::operator!=(const op_map &m) const { return !(*this == m); }
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const op_map &m) {
  if (m != op_map::no_map)
    return os << "map:" << m.name << "::" << m.from << "-->" << m.to
              << " dim: " << m.dim;
  return os << "map:OP_ID";
}
const op_map op_map::no_map("", "", 0, "");

//_________________________________USER_FUNC__________________________________
UserFuncData::UserFuncData(const clang::FunctionDecl *funcD,
                           const clang::SourceManager *sm)
    : functionDecl(decl2str(funcD, sm)), funcName(funcD->getNameAsString()) {
  isInlineSpecified = funcD->isInlineSpecified();
  path = funcD->getLocStart().printToString(*sm);
  path = path.substr(0, path.find(":"));
  for (size_t i = 0; i < funcD->getNumParams(); ++i) {
    paramNames.push_back(funcD->getParamDecl(i)->getNameAsString());
  }
}
std::string UserFuncData::getInlinedFuncDecl() const {
  return (isInlineSpecified ? "" : "inline ") + functionDecl;
}

//___________________________________OP_ARG___________________________________
DummyOPArgv2::DummyOPArgv2(const clang::VarDecl *dat, int _idx,
                           const std::string &_map, size_t _dim, std::string _type,
                           OP_accs_type _accs)
    : opDat(dat->getNameAsString()), idx(_idx), opMap(_map), dim(_dim),
      type(_type), accs(_accs), isGBL(false) {}

DummyOPArgv2::DummyOPArgv2(const clang::VarDecl *dat, size_t _dim,
                           std::string _type, OP_accs_type _accs)
    : opDat(dat->getNameAsString()), idx(0), opMap(""), dim(_dim),
      type(_type), accs(_accs), isGBL(true) {}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DummyOPArgv2 &arg) {
  os << "op_arg" << (arg.isGBL ? "_gbl" : "") << ":\n\t"
     << "op_dat: " << arg.opDat << "\n\t";
  if (!arg.isGBL) {
    if (arg.opMap != "") { // indirect argument
      os << "map_idx: " << arg.idx << "\n\t";
    }
    os << arg.opMap << "\n\t";
  }

  return os << "dim: " << arg.dim << "\n\ttype: " << arg.type
            << "\n\taccess: " << arg.accs << "\n";
}
bool DummyOPArgv2::isDirect() const { return opMap == ""; }

std::string DummyOPArgv2::getArgCall(int argIdx, std::string mapStr) const {
  std::string data = "(" + type + "*)arg" + std::to_string(argIdx) + ".data";
  if (isGBL) {
    return data;
  }
  return "&(" + data + ")[" + std::to_string(dim) + "*" + mapStr + "]";
}

bool DummyOPArgv2::isReduction() const {
  return isGBL && (accs == OP_INC || accs == OP_MAX || accs == OP_MIN);
}

//__________________________________PAR_LOOP__________________________________

size_t DummyParLoop::numLoops = 0;

DummyParLoop::DummyParLoop(const clang::FunctionDecl *_function,
                           const clang::SourceManager *sm, std::string _name,
                           std::vector<OPArg> _args)
    : loopId(numLoops++), function(_function, sm), name(_name), args(_args) {
  std::map<std::string, int> datToInd;
  std::map<std::string, std::map<int, int>> mapidxToInd;
  std::map<std::string, int> map2ind;
  int mapidxs = 0;
  int nummaps = -1;
  for (size_t i = 0; i < args.size(); ++i) {
    OPArg &arg = args[i];
    if (arg.isDirect() || arg.isGBL) {
      dataIdxs.push_back(-1);
      mapIdxs.push_back(-1);
      arg2map.push_back(-1);
    } else {
      auto it = datToInd.find(arg.opDat);
      if (it != datToInd.end()) {
        dataIdxs.push_back(it->second);
      } else {
        dat2argIdxs.push_back(i);
        ninds = datToInd.size();
        dataIdxs.push_back(ninds);
        datToInd[arg.opDat] = ninds;
        ninds++;
      }
      auto mapIt = mapidxToInd.find(arg.opMap);
      if (mapIt == mapidxToInd.end()) {
        std::map<int, int> tmp;
        mapIdxs.push_back(mapidxs);
        map2argIdxs.push_back(i);
        arg2map.push_back(++nummaps);
        map2ind[arg.opMap] = nummaps;
        tmp[arg.idx] = mapidxs++;
        mapidxToInd[arg.opMap] = tmp;
      } else {
        arg2map.push_back(map2ind[arg.opMap]);
        auto idxit = mapIt->second.find(arg.idx);
        if (idxit == mapIt->second.end()) {
          mapIdxs.push_back(mapidxs);
          mapIt->second[arg.idx] = mapidxs++;
        } else {
          mapIdxs.push_back(idxit->second);
        }
      }
    }
  }
}

bool DummyParLoop::isDirect() const {
  return std::all_of(args.begin(), args.end(),
                     [](const OPArg &a) { return a.isDirect(); });
}

std::string DummyParLoop::getName() const { return name; }
std::string DummyParLoop::getFuncText() const { return function.functionDecl; }

size_t DummyParLoop::getLoopID() const { return loopId; }

std::string DummyParLoop::getUserFuncInc() const {
  return function.getInlinedFuncDecl();
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

std::string DummyParLoop::getFuncCall() const {
  std::string funcCall = "";
  llvm::raw_string_ostream ss(funcCall);
  ss << function.funcName << "(";
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].isDirect()) {
      ss << args[i].getArgCall(i, "n");
    } else {
      ss << args[i].getArgCall(dat2argIdxs[dataIdxs[i]],
                               ("map" + std::to_string(mapIdxs[i]) + "idx"));
    }
    ss << ",";
  }
  ss.str();

  return funcCall.substr(0, funcCall.length() - 1) + ");";
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
  std::vector<int> mapinds(args.size(), -1);
  std::map<std::string, int> mapToArg;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].isDirect() || args[i].isGBL) {
      continue;
    }
    if (mapinds[mapIdxs[i]] == -1) {
      mapinds[mapIdxs[i]] = i;
      ss << "int map" << mapIdxs[i] << "idx = arg" << map2argIdxs[arg2map[i]]
         << ".map_data[n * arg" << map2argIdxs[arg2map[i]] << ".map->dim + "
         << args[i].idx << "];\n";
    }
  }
  return ss.str();
}

UserFuncData DummyParLoop::getUserFuncInfo() const { return function; }

void DummyParLoop::dumpFuncTextTo(std::string path) const {
  std::ofstream os(path);
  os << function.functionDecl;
  os.close();
}

void OP2Application::setName(std::string name) { applicationName = name; }

std::vector<ParLoop> &OP2Application::getParLoops() { return loops; }
const std::vector<ParLoop> &OP2Application::getParLoops() const {
  return loops;
}

} // namespace OP2
