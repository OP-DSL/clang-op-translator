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
OPArg::OPArg(size_t idx, std::string datName, size_t datDim, std::string type,
             OP_accs_type accs, OPArg::OPArgKind kind, int mapIdx,
             std::string mapName, bool opt)
    : argIdx(idx), opDat(datName), dim(datDim), type(type), accs(accs),
      kind(kind), mapIdx(mapIdx), opMap(mapName), optional(opt) {}

OPArg::OPArg(size_t idx)
    : argIdx(idx), opDat(""), dim(1ul), type(""), accs(OP_READ),
      kind(OPArg::OP_IDX), mapIdx(-1), opMap(""), optional(false) {}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const OPArg &arg) {
  // TODO update or remove

  os << "arg: " << arg.kind;
  if (arg.kind == OPArg::OP_IDX)
    return os;
  os << " " << arg.opDat << ", dim: " << arg.dim << ", " << arg.type << ", "
     << arg.accs;
  if (arg.kind == OPArg::OPArgKind::OP_DAT) {
    os << ", idx: " << arg.mapIdx << ", map: " << arg.opMap
       << (arg.optional ? ", opt" : "");
  }
  return os;
}
bool OPArg::isDirect() const { return opMap == ""; }

bool OPArg::isReduction() const { return OPArgKind::OP_REDUCE == kind; }

//__________________________________PAR_LOOP__________________________________

ParLoop::ParLoop(UserFuncData _userFuncData, std::string _name,
                 std::vector<OPArg> _args, OPLoopKind kind)
    : loopId(-1), function(_userFuncData), name(_name), args(_args),
      loopKind(kind) {}

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
  for (size_t i = 0; i < args.size(); ++i) {
    os << ", op_arg arg" << i;
  }
  os << ")";
  return os.str();
}

UserFuncData ParLoop::getUserFuncInfo() const { return function; }

void ParLoop::prettyPrint(llvm::raw_ostream &o) const {
  o << loopId << " " << loopKind << " par_loop " << name << '\n';
  o << function.funcName << ": " << function.path << '\n';
  for (const auto &arg : args) {
    o << arg << '\n';
  }
}

OPLoopKind ParLoop::getKind() const { return loopKind; }

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
