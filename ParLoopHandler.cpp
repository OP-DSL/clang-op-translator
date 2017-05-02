#include "ParLoopHandler.h"
#include "utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/Preprocessor.h"
#include <llvm/Support/Debug.h>
#include <sstream>

namespace OP2 {

void ParLoopHandler::run(const matchers::MatchFinder::MatchResult &Result) {
  const clang::CallExpr *function =
      Result.Nodes.getNodeAs<clang::CallExpr>("par_loop");
  if (function->getNumArgs() < 3) {
    reportDiagnostic(*Result.Context, function,
                     "not enough arguments to op_par_loop");
  }
  const clang::Expr *str_arg = function->getArg(1);
  const clang::StringLiteral *name = getAsStringLiteral(str_arg);
  const auto *fExpr =
      llvm::dyn_cast<clang::DeclRefExpr>(function->getArg(0)->IgnoreCasts());
  const auto *fDecl =
      llvm::dyn_cast<clang::FunctionDecl>(fExpr->getFoundDecl());
  if (!fDecl) {
    reportDiagnostic(*Result.Context, function->getArg(0), "Must be a function pointer");
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
    ss << ", op_dat";
  }
  ss << ");\n\n";
  Rewriter.InsertTextBefore(parent->getLocStart(), ss.str());

  // Reset the stringstream;
  ss.str({});
  ss << "_" << name->getString().str();
  Rewriter.InsertTextAfter(function->getLocStart().getLocWithOffset(11),
                           ss.str());
  Rewriter.RemoveText(
      {function->getArg(0)->getLocStart(), function->getArg(1)->getLocStart()});

  ss.str({});
  ss << "op_par_loop_" << name->getString().str();

  auto newFile = createCompilerInstance();
  auto TU = newFile->getASTContext().getTranslationUnitDecl();
  clang::QualType FTy = newFile->getASTContext().getFunctionType(
      newFile->getASTContext().IntTy, {},
      clang::FunctionProtoType::ExtProtoInfo());
  clang::FunctionDecl *FD = clang::FunctionDecl::Create(
      newFile->getASTContext(), TU, {}, {},
      clang::DeclarationName(
          &newFile->getPreprocessor().getIdentifierTable().get(ss.str())),
      FTy, nullptr, clang::SC_None, false);
  clang::FunctionDecl *oldFun = clang::FunctionDecl::Create(
      newFile->getASTContext(), TU, {}, {},
      clang::DeclarationName(
          &newFile->getPreprocessor().getIdentifierTable().get(fDecl->getName())),
      function->getType(), nullptr, clang::SC_None, false);
  oldFun->setBody(fDecl->getBody());
  ss << ".cpp";
  TU->addDecl(oldFun);
  TU->addDecl(FD);
  std::error_code ec;
  llvm::raw_fd_ostream outfile{ss.str(), ec,
                               llvm::sys::fs::F_Text | llvm::sys::fs::F_RW};
  TU->print(outfile);
}

}
