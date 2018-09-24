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

//_________________________________USER_FUNC__________________________________
UserFuncData::UserFuncData(std::string _fDecl, std::string _fName,
                           bool _isinline, std::string _path,
                           std::vector<std::string> _paramNames)
    : functionDecl(_fDecl), funcName(_fName), isInlineSpecified(_isinline),
      path(_path), paramNames(_paramNames) {}

std::string UserFuncData::getInlinedFuncDecl() const {
  return (isInlineSpecified ? "" : "inline ") + functionDecl;
}

//___________________________________OP_ARG___________________________________
OPArg::OPArg(int _argIdx, std::string dat, int _idx, const std::string &_map,
             size_t _dim, std::string _type, OP_accs_type _accs)
    : argIdx(_argIdx), opDat(dat), idx(_idx), opMap(_map), dim(_dim),
      type(_type), accs(_accs), isGBL(false) {}

OPArg::OPArg(int _argIdx, std::string dat, size_t _dim, std::string _type,
             OP_accs_type _accs)
    : argIdx(_argIdx), opDat(dat), idx(0), opMap(""), dim(_dim), type(_type),
      accs(_accs), isGBL(true) {}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const OPArg &arg) {
  os << "op_arg" << (arg.isGBL ? "_gbl(" : "(dat: ") << arg.opDat << ", ";
  if (!arg.isGBL) {
    os << "idx: " << arg.idx << ", map: " << arg.opMap << ", ";
  }
  return os << "dim: " << arg.dim << ", type: " << arg.type
            << ", accs: " << arg.accs << ")";
}
bool OPArg::isDirect() const { return opMap == ""; }

bool OPArg::isReduction() const {
  return isGBL && (accs == OP_INC || accs == OP_MAX || accs == OP_MIN);
}

//__________________________________PAR_LOOP__________________________________

ParLoop::ParLoop(UserFuncData _userFuncData, std::string _name,
                 std::vector<OPArg> _args)
    : loopId(-1), function(_userFuncData), name(_name), args(_args) {}

void ParLoop::generateID() {
  if (loopId == -1)
    loopId = numLoops++;
}

bool ParLoop::isDirect() const {
  return std::all_of(args.begin(), args.end(),
                     [](const OPArg &a) { return a.isDirect(); });
}

std::string ParLoop::getName() const { return name; }

size_t ParLoop::getLoopID() const { return loopId; }

std::string ParLoop::getParLoopDef() const {
  std::string param = "void par_loop_" + name + "(const char *name, op_set set";
  llvm::raw_string_ostream os(param);
  for (size_t i = 2; i < args.size(); ++i) {
    os << ", op_arg arg" << i - 2;
  }
  os << ")";
  return os.str();
}

UserFuncData ParLoop::getUserFuncInfo() const { return function; }

//________________________________OPAPPLICATION_______________________________
void OPApplication::setName(std::string name) { applicationName = name; }

std::vector<ParLoop> &OPApplication::getParLoops() { return loops; }
const std::vector<ParLoop> &OPApplication::getParLoops() const { return loops; }

bool OPApplication::addParLoop(ParLoop newLoop) {
  for (const ParLoop &loop : loops) {
    if (newLoop.getName() == loop.getName()) {
      return false;
    }
  }
  newLoop.generateID();
  loops.push_back(newLoop);
  return true;
}

} // namespace OP2
