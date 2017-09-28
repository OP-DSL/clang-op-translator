#include "BaseKernelHandler.h"
#include "../utils.h"

namespace OP2 {
using namespace clang::ast_matchers;

const DeclarationMatcher BaseKernelHandler::parLoopDeclMatcher =
    functionDecl(hasName("op_par_loop_skeleton")).bind("par_loop_decl");
const DeclarationMatcher BaseKernelHandler::nargsMatcher =
    varDecl(hasType(isInteger()), hasName("nargs")).bind("nargs_decl");

BaseKernelHandler::BaseKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

void BaseKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!handleParLoopDecl(Result))
    return; // if successfully handled return
  if (!handleNargsDecl(Result))
    return;
}

int BaseKernelHandler::handleParLoopDecl(
    const MatchFinder::MatchResult &Result) {
  const clang::FunctionDecl *function =
      Result.Nodes.getNodeAs<clang::FunctionDecl>("par_loop_decl");
  if (!function)
    return 1; // We shouldn't handle this match
  clang::SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(function->getLocStart(), sm);
  // replace skeleton to the name of the loop
  size_t nameoffset = std::string("void op_par_loop_").length();
  size_t length = std::string("skeleton").length();
  clang::tooling::Replacement funcNameRep(
      *sm, function->getLocStart().getLocWithOffset(nameoffset), length,
      loop.getName());
  if (llvm::Error err = (*Replace)[filename].add(funcNameRep)) {
    // TODO diagnostics..
    llvm::errs() << "Function name replacement failed in: " << filename << "\n";
  }
  // add op_args to the parameter list
  std::string arg_repl = "";
  llvm::raw_string_ostream os(arg_repl);
  for (size_t i = 1; i < loop.getNumArgs(); ++i) {
    os << ", op_arg arg" << i;
  }

  llvm::outs() << loop.getNumArgs() << "\n";
  if (loop.getNumArgs() > 1) {
    clang::tooling::Replacement funcArgRep(
        *sm,
        function->getParamDecl(function->getNumParams() - 1)
            ->getLocEnd()
            .getLocWithOffset(4 /*length of arg0*/),
        0, os.str());
    if (llvm::Error err = (*Replace)[filename].add(funcArgRep)) {
      // TODO diagnostics..
      llvm::errs() << "Function arg addition failed in: " << filename << "\n";
    }
  }
  return 0;
}

int BaseKernelHandler::handleNargsDecl(const MatchFinder::MatchResult &Result) {
  const clang::VarDecl *nargsDecl =
      Result.Nodes.getNodeAs<clang::VarDecl>("nargs_decl");
  if (!nargsDecl)
    return 1; // We shouldn't handle this match
  clang::SourceManager *sm = Result.SourceManager;
  if (!sm->isWrittenInMainFile(nargsDecl->getLocStart()))
    return 0;
  std::string filename = getFileNameFromSourceLoc(nargsDecl->getLocStart(), sm);

  // change value of nargs
  if (loop.getNumArgs() <= 1)
    return 0; // there is no need to change
  clang::tooling::Replacement repl(*sm, nargsDecl->getLocEnd(),
                                   1 /*len of 1..*/,
                                   std::to_string(loop.getNumArgs()));
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set value of nargs failed in: " << filename << "\n";
  }

  return 0;
}

} // end of namespace OP2
