#include "AppFileTransformations.hpp"
#include "AppFileRefactoringTool.hpp"
#include "core/utils.h"
#include <clang/Frontend/CompilerInstance.h>

namespace OP2 {
//_____________________________PARLOOPDECLARATOR_______________________________
ParLoopDeclarator::ParLoopDeclarator(AppFileRefactoringTool &tool)
    : tool(tool) {}

bool ParLoopDeclarator::handleBeginSource(clang::CompilerInstance &CI) {
  CI.getPreprocessor().addPPCallbacks(
      std::make_unique<IncludeFinderPPCallback>(&CI, this));
  return true;
}

void ParLoopDeclarator::handleEndSource() {
  assert(!fileName.empty() && replRange != clang::SourceRange() && SM);
  clang::tooling::Replacement repl(
      *SM, clang::CharSourceRange(replRange, false), functionDeclarations);
  tool.addReplacementTo(fileName, repl, "par_loop predeclarations");
  functionDeclarations = "";
  replRange = clang::SourceRange();
  fileName = "";
}

void ParLoopDeclarator::addFunction(const std::string &funcDeclaration) {
  if (functionDeclarations.find(funcDeclaration) == std::string::npos)
    functionDeclarations += funcDeclaration;
}
void ParLoopDeclarator::setCurrentFile(std::string fName, clang::SourceRange sr,
                                       clang::SourceManager *SM,
                                       const std::string &matchFileName) {
  if (fileName.empty()) {
    fileName = std::move(fName);
    replRange = sr;
    this->SM = SM;
    if (matchFileName == "op2_seq.h") {
      functionDeclarations = "#include \"op_lib_cpp.h\"\n";
    } else {
      functionDeclarations = "#include \"ops_lib_cpp.h\"\n";
    }
  } else {
    llvm::errs()
        << "Warning multiple #include \"op_seq.h\" in the processed file\n";
  }
}

ParLoopDeclarator::IncludeFinderPPCallback::IncludeFinderPPCallback(
    clang::CompilerInstance *CI, ParLoopDeclarator *callback)
    : CI(CI), callback(callback) {}

void ParLoopDeclarator::IncludeFinderPPCallback::InclusionDirective(
    clang::SourceLocation HashLoc, const clang::Token &,
    clang::StringRef fileName, bool, clang::CharSourceRange FilenameRange,
    const clang::FileEntry *, clang::StringRef, clang::StringRef,
    const clang::Module *, clang::SrcMgr::CharacteristicKind) {
  if ((fileName == "op_seq.h" || fileName == "ops_seq.h") &&
      CI->getSourceManager().isInMainFile(HashLoc)) {
    callback->setCurrentFile(
        CI->getSourceManager().getFilename(HashLoc).str(),
        clang::SourceRange(HashLoc, FilenameRange.getEnd().getLocWithOffset(2)),
        &CI->getSourceManager(), fileName);
  }
}

//_________________________PARLOOPCALLREPLACEOPERATION_________________________

ParloopCallReplaceOperation::ParloopCallReplaceOperation(
    OPApplication &app, ParLoopDeclarator &decl, AppFileRefactoringTool &tool)
    : app(app), parLoopDeclarator(decl), tool(tool) {}

void ParloopCallReplaceOperation::
operator()(const matchers::MatchFinder::MatchResult &Result) const {
  const auto *parLoopCall = Result.Nodes.getNodeAs<clang::CallExpr>("par_loop");
  const auto *fDecl =
      getExprAsDecl<clang::FunctionDecl>(parLoopCall->getArg(0));
  std::string fname = fDecl->getNameAsString();
  auto loop = std::find_if(
      app.getParLoops().begin(), app.getParLoops().end(),
      [&fname](const auto &loop) { return loop.getName() == fname; });
  if (loop != app.getParLoops().end()) {
    parLoopDeclarator.addFunction(loop->getParLoopDef() + ";\n\n");

    std::string start = "op_par_loop";
    if (loop->getKind() == OPS) {
      start = "ops_par_loop";
    }
    clang::SourceRange replRange(parLoopCall->getBeginLoc(),
                                 parLoopCall->getArg(1)->getBeginLoc());
    clang::SourceManager *SM = Result.SourceManager;
    clang::tooling::Replacement func_Rep(
        *SM, clang::CharSourceRange(replRange, false),
        start + "_" + fname + "(");
    tool.addReplacementTo(SM->getFilename(parLoopCall->getBeginLoc()), func_Rep,
                          "func_call");
  }
}

//________________________OPCONSTDECLAREREPLACEOPERATION_______________________

OPConstDeclarationReplaceOperation::OPConstDeclarationReplaceOperation(
    AppFileRefactoringTool &_tool)
    : tool(_tool) {}

void OPConstDeclarationReplaceOperation::
operator()(const matchers::MatchFinder::MatchResult &Result) const {
  const auto *declConstCall =
      Result.Nodes.getNodeAs<clang::CallExpr>("decl_const");
  const auto *fDecl = declConstCall->getDirectCallee();

  std::string replS = fDecl->getNameAsString() + "2(";
  if (declConstCall->getNumArgs() == 3) {
    // dim, type, var -- missing name in first arg
    replS += '"' +
             getExprAsDecl<clang::VarDecl>(declConstCall->getArg(2))
                 ->getNameAsString() +
             "\",";
  }

  clang::SourceRange replRange(declConstCall->getBeginLoc(),
                               declConstCall->getArg(0)->getBeginLoc());
  clang::SourceManager *SM = Result.SourceManager;
  clang::tooling::Replacement repl(
      *SM, clang::CharSourceRange(replRange, false), replS);
  tool.addReplacementTo(SM->getFilename(declConstCall->getBeginLoc()), repl,
                        "func_call");
}

} // namespace OP2
