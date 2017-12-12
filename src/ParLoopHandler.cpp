#include "ParLoopHandler.h"
#include "core/utils.h"
#include <sstream>

namespace OP2 {

void addOPArgToVector(const clang::Expr *argExpr, std::vector<OPArg> &args,
                      OP2Application &app) {
  const clang::Stmt *argStmt = llvm::dyn_cast<clang::Stmt>(argExpr);
  // ugly solution to get the op_arg_dat callExpr from from AST..
  while (!llvm::isa<clang::CallExpr>(argStmt)) {
    unsigned num_childs = 0;
    const clang::Stmt *parentStmt = argStmt;
    for (const clang::Stmt *child : parentStmt->children()) {
      num_childs++;
      argStmt = child;
    }
    assert(num_childs == 1);
  }
  const clang::CallExpr *argCallExpr = llvm::dyn_cast<clang::CallExpr>(argStmt);
  // argCallExpr->dump();
  std::string fname =
      llvm::dyn_cast<clang::NamedDecl>(argCallExpr->getCalleeDecl())
          ->getNameAsString();
  bool isGBL = fname == "op_arg_gbl";
  if (!isGBL && fname != "op_arg_dat") {
    llvm::errs() << "Unknown arg declaration: " << fname << "\n";
    return;
  }
  const clang::VarDecl *opDat =
      getExprAsDecl<clang::VarDecl>(argCallExpr->getArg(0));
  int idx = -2;
  std::string opMap = "";
  if (!isGBL) {
    idx = getIntValFromExpr(argCallExpr->getArg(1)->IgnoreCasts());
    if (idx != -1) {
      opMap = getExprAsDecl<clang::VarDecl>(argCallExpr->getArg(2))
                  ->getNameAsString();
    }
  } else {
    llvm::outs() << opDat << "\n";
  }
  size_t dim = getIntValFromExpr(
      argCallExpr->getArg(3 - (isGBL ? 2 : 0))->IgnoreCasts());
  std::string type =
      getAsStringLiteral(argCallExpr->getArg(4 - (isGBL ? 2 : 0)))
          ->getString()
          .str();
  OP_accs_type accs = OP_accs_type(getIntValFromExpr(
      argCallExpr->getArg(5 - (isGBL ? 2 : 0))->IgnoreCasts()));

  if (!isGBL) {
    if (idx != -1) {
      args.push_back(
          OPArg(opDat, idx, app.mappings.find(opMap)->second, dim, type, accs));
    } else {
      args.push_back(OPArg(opDat, idx, op_map::no_map, dim, type, accs));
    }
  } else {
    args.push_back(OPArg(opDat, dim, type, accs));
  }
}

void ParLoopHandler::parseFunctionDecl(const clang::CallExpr *parloopExpr,
                                       const clang::SourceManager *SM) {
  std::vector<OPArg> args;
  const clang::FunctionDecl *fDecl =
      getExprAsDecl<clang::FunctionDecl>(parloopExpr->getArg(0)->IgnoreCasts());
  const clang::VarDecl *setDecl =
      getExprAsDecl<clang::VarDecl>(parloopExpr->getArg(2)->IgnoreCasts());
  std::string data;
  llvm::raw_string_ostream parLoopDataSS(data);
  parLoopDataSS << "------------------------------\n";
  std::string name =
      getAsStringLiteral(parloopExpr->getArg(1))->getString().str();
  parLoopDataSS << "name: " << name << "\nfunction def: \n";
  clang::SourceRange fDeclSR = fDecl->getSourceRange();
  parLoopDataSS << "  starts at: " << fDeclSR.getBegin().printToString(*SM)
                << "\n";
  parLoopDataSS << "  ends at: " << fDeclSR.getEnd().printToString(*SM) << "\n";
  parLoopDataSS << "iteration set: " << setDecl->getNameAsString() << "\n";

  for (unsigned arg_ind = 3; arg_ind < parloopExpr->getNumArgs(); ++arg_ind) {
    parLoopDataSS << "arg" << arg_ind - 3 << ":\n";
    addOPArgToVector(parloopExpr->getArg(arg_ind), args, app);
    parLoopDataSS << args.back();
  }
  app.getParLoops().push_back(ParLoop(fDecl, SM, name, args));
  llvm::outs() << parLoopDataSS.str();
}

void ParLoopHandler::run(const matchers::MatchFinder::MatchResult &Result) {
  const clang::CallExpr *function =
      Result.Nodes.getNodeAs<clang::CallExpr>("par_loop");
  if (function->getNumArgs() < 3) {
    reportDiagnostic(*Result.Context, function,
                     "not enough arguments to op_par_loop");
  }
  const clang::Expr *str_arg = function->getArg(1);
  const clang::StringLiteral *name = getAsStringLiteral(str_arg);
  const clang::FunctionDecl *fDecl =
      getExprAsDecl<clang::FunctionDecl>(function->getArg(0));

  if (!fDecl) {
    reportDiagnostic(*Result.Context, function->getArg(0),
                     "Must be a function pointer");
    return;
  } else if (!fDecl->hasBody()) {
    reportDiagnostic(
        *Result.Context, function->getArg(0),
        "body must be available at the point of an op_par_loop call");
  }

  // If the second argument isn't a string literal, issue an error
  if (!name) {
    reportDiagnostic(*Result.Context, str_arg,
                     "op_par_loop called with non-string literal kernel name",
                     clang::DiagnosticsEngine::Warning);
  }

  const clang::FunctionDecl *parent =
      findParent<clang::FunctionDecl>(*function, *Result.Context);
  std::stringstream ss;
  ss << "void op_par_loop_" << name->getString().str()
     << "(const char*, op_set";
  for (unsigned i = 0; i < function->getNumArgs() - 3; ++i) {
    ss << ", op_arg";
  }
  ss << ");\n\n";
  std::string func_signature = ss.str();

  // Reset the stringstream;
  ss.str({});
  ss << "_" << name->getString().str() << "(";

  // get the current filename
  clang::SourceManager *sourceManager = Result.SourceManager;
  const std::string fname =
      getFileNameFromSourceLoc(function->getLocStart(), sourceManager);

  clang::tooling::Replacements &Rpls = (*Replace)[fname];
  clang::tooling::Replacement Rep(*sourceManager, parent->getLocStart(), 0,
                                  func_signature);

  (*Replace)[fname] = Rpls.merge(clang::tooling::Replacements(Rep));
  // Add replacement for func call
  unsigned length =
      sourceManager->getFileOffset(function->getArg(1)->getLocStart()) -
      sourceManager->getFileOffset(
          function->getLocStart().getLocWithOffset(11));
  clang::tooling::Replacement func_Rep(
      *sourceManager, function->getLocStart().getLocWithOffset(11), length,
      ss.str());

  llvm::Error err = (*Replace)[fname].add(func_Rep);
  if (err) { // TODO proper error checking
    llvm::outs()
        << "Some Error occured during adding replacement for func_call\n";
  }
  // End adding Replacements

  // parse func decl test
  parseFunctionDecl(function, Result.SourceManager);
}

} // namespace OP2
