#include "ParLoopHandler.hpp"
#include "OP2RefactoringTool.hpp"
#include "core/utils.h"
#include <clang/Tooling/Tooling.h>
#include <sstream>

namespace OP2 {

//_____________________________PARLOOPDECLARATOR_______________________________
OPParLoopDeclarator::OPParLoopDeclarator(OP2RefactoringTool &tool)
    : functionDeclarations("#include \"op_lib_cpp.h\"\n"), tool(tool) {}

bool OPParLoopDeclarator::handleBeginSource(clang::CompilerInstance &CI) {
  std::unique_ptr<IncludeFinderPPCallback> find_includes_callback(
      new IncludeFinderPPCallback(&CI, this));
  clang::Preprocessor &pp = CI.getPreprocessor();
  pp.addPPCallbacks(std::move(find_includes_callback));
  return true;
}

void OPParLoopDeclarator::handleEndSource() {
  assert(fileName != "" && replRange != clang::SourceRange() && SM);
  llvm::outs() << functionDeclarations << "\n";
  clang::tooling::Replacement repl(
      *SM, clang::CharSourceRange(replRange, false), functionDeclarations);
  tool.addReplacementTo(fileName, repl, "op_lib_cpp.h include");
  functionDeclarations = "#include \"op_lib_cpp.h\"\n";
  replRange = clang::SourceRange();
  fileName = "";
}

void OPParLoopDeclarator::addFunction(std::string funcDeclaration) {
  functionDeclarations += funcDeclaration;
}
void OPParLoopDeclarator::setCurrentFile(std::string fName,
                                         clang::SourceRange sr,
                                         clang::SourceManager *SM) {
  if (fileName == "") {
    fileName = fName;
    replRange = sr;
    this->SM = SM;
  } else {
    llvm::errs()
        << "Warning multiple #include \"op_seq.h\" in the processed file\n";
  }
}
OPParLoopDeclarator::IncludeFinderPPCallback::IncludeFinderPPCallback(
    clang::CompilerInstance *CI, OPParLoopDeclarator *callback)
    : CI(CI), callback(callback) {}
void OPParLoopDeclarator::IncludeFinderPPCallback::InclusionDirective(
    clang::SourceLocation HashLoc, const clang::Token &, StringRef FileName,
    bool, clang::CharSourceRange FilenameRange, const clang::FileEntry *,
    StringRef, StringRef, const clang::Module *) {

  if (FileName == "op_seq.h" && CI->getSourceManager().isInMainFile(HashLoc)) {
    callback->setCurrentFile(
        CI->getSourceManager().getFilename(HashLoc).str(),
        clang::SourceRange(HashLoc, FilenameRange.getEnd().getLocWithOffset(2)),
        &CI->getSourceManager());
  }
}

//_______________________________PARLOOPHANDLER________________________________
void addOPArgToVector(const clang::Expr *argExpr, std::vector<OPArg> &args,
                      const clang::SourceManager *SM) {
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
  std::string opDat =
      getSourceAsString(argCallExpr->getArg(0)->getSourceRange(), SM);
  int idx = -2;
  std::string opMap = "";
  if (!isGBL) {
    idx = getIntValFromExpr(argCallExpr->getArg(1)->IgnoreCasts());
    if (idx != -1) {
      opMap = getExprAsDecl<clang::VarDecl>(argCallExpr->getArg(2))
                  ->getNameAsString();
    }
  }
  size_t argIdx = 3 - (isGBL ? 2 : 0);
  size_t dim = getIntValFromExpr(argCallExpr->getArg(argIdx++)->IgnoreCasts());
  std::string type =
      getAsStringLiteral(argCallExpr->getArg(argIdx++))->getString().str();
  OP_accs_type accs = OP_accs_type(
      getIntValFromExpr(argCallExpr->getArg(argIdx)->IgnoreCasts()));

  if (!isGBL) {
    args.push_back(OPArg(opDat, idx, opMap, dim, type, accs));
  } else {
    args.push_back(OPArg(opDat, dim, type, accs));
  }
}

ParLoopHandler::ParLoopHandler(OP2RefactoringTool &tool, OP2Application &app,
                               OPParLoopDeclarator &declarator)
    : tool(tool), app(app), declarator(declarator) {}

void ParLoopHandler::parseFunctionDecl(const clang::CallExpr *parloopExpr,
                                       const clang::SourceManager *SM) {
  std::vector<OPArg> args;
  const clang::FunctionDecl *fDecl =
      getExprAsDecl<clang::FunctionDecl>(parloopExpr->getArg(0)->IgnoreCasts());
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
  parLoopDataSS << "iteration set: "
                << getSourceAsString(parloopExpr->getArg(2)->getSourceRange(),
                                     SM)
                << "\n";
  for (unsigned arg_ind = 3; arg_ind < parloopExpr->getNumArgs(); ++arg_ind) {
    parLoopDataSS << "arg" << arg_ind - 3 << ":\n";
    addOPArgToVector(parloopExpr->getArg(arg_ind), args, SM);
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

  declarator.addFunction(func_signature);

  // Reset the stringstream;
  ss.str({});
  ss << "_" << name->getString().str() << "(";

  // get the current filename
  clang::SourceManager *sourceManager = Result.SourceManager;
  const std::string fname =
      getFileNameFromSourceLoc(function->getLocStart(), sourceManager);

  //  clang::tooling::Replacements &Rpls = (*Replace)[fname];
  clang::tooling::Replacement Rep(*sourceManager, parent->getLocStart(), 0,
                                  func_signature);

  //  (*Replace)[fname] = Rpls.merge(clang::tooling::Replacements(Rep));
  // Add replacement for func call
  unsigned length =
      sourceManager->getFileOffset(function->getArg(1)->getLocStart()) -
      sourceManager->getFileOffset(
          function->getLocStart().getLocWithOffset(11));
  clang::tooling::Replacement func_Rep(
      *sourceManager, function->getLocStart().getLocWithOffset(11), length,
      ss.str());

  tool.addReplacementTo(fname, func_Rep, "func_call");
  // End adding Replacements

  // parse func decl test
  parseFunctionDecl(function, Result.SourceManager);
}

} // namespace OP2
