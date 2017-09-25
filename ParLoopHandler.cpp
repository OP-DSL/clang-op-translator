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

void ParLoopHandler::parseFunctionDecl(const clang::CallExpr *parloopExpr,
                                       const clang::SourceManager *SM) {
  std::stringstream parLoopDataSS;
  parLoopDataSS << "------------------------------\n";
  const clang::StringLiteral *name = getAsStringLiteral(parloopExpr->getArg(1));
  parLoopDataSS << "name: " << name->getString().str() << "\n";
  
  parLoopDataSS << "function def: \n";
  const clang::DeclRefExpr *expr =
      llvm::dyn_cast<clang::DeclRefExpr>(parloopExpr->getArg(0)->IgnoreCasts());
  const clang::FunctionDecl *fDecl =
      llvm::dyn_cast<clang::FunctionDecl>(expr->getFoundDecl());
  clang::SourceRange fDeclSR = fDecl->getSourceRange();
  parLoopDataSS << "  starts at: " << fDeclSR.getBegin().printToString(*SM)
             << "\n";
  parLoopDataSS << "  ends at: " << fDeclSR.getEnd().printToString(*SM) << "\n";
  
  const clang::DeclRefExpr *setDeclRef =
      llvm::dyn_cast<clang::DeclRefExpr>(parloopExpr->getArg(2)->IgnoreCasts());
  const clang::VarDecl *setDecl =
      llvm::dyn_cast<clang::VarDecl>(setDeclRef->getFoundDecl());
  parLoopDataSS << "iteration set: " << setDecl->getNameAsString() << "\n";

  for (unsigned arg_ind = 3; arg_ind < parloopExpr->getNumArgs(); ++arg_ind) {
    parLoopDataSS << "arg" << arg_ind - 3 << ":\n";

    const clang::Expr *argExpr = parloopExpr->getArg(arg_ind);
    const clang::Stmt *argStmt = llvm::dyn_cast<clang::Stmt>(argExpr);
    //ugly solution to get the op_arg_dat callExpr from from AST..
    while (!llvm::isa<clang::CallExpr>(argStmt)) {
      unsigned num_childs = 0;
      const clang::Stmt *parentStmt = argStmt;
      for (const clang::Stmt *child : parentStmt->children()) {
        num_childs++;
        argStmt = child;
      }
      assert(num_childs == 1);
    }
    const clang::CallExpr *argCallExpr =
        llvm::dyn_cast<clang::CallExpr>(argStmt);
    argCallExpr->dump();
    std::string fname = llvm::dyn_cast<clang::NamedDecl>(argCallExpr->getCalleeDecl())->getName().str();
    if(fname == "op_arg_gbl"){ //TODO
      parLoopDataSS << fname << "\n";
      continue;
    }

    parLoopDataSS << argCallExpr->getType().getAsString() << "\n";
    const clang::VarDecl *opDat = getExprAsVarDecl(argCallExpr->getArg(0));
    parLoopDataSS << "  op_dat: " << opDat->getType().getAsString() << " "
               << opDat->getName().str() << " "
               << opDat->getDefinition(opDat->getASTContext())
                      ->getLocation()
                      .printToString(*SM)
               << "\n";

    const clang::Expr *idxExpr = argCallExpr->getArg(1)->IgnoreCasts(); 
    parLoopDataSS << "  idx: " << getIntValFromExpr(idxExpr)
               << "\n";

    const clang::VarDecl *opMap = getExprAsVarDecl(argCallExpr->getArg(2));

    parLoopDataSS << "  map: ";
    if(opMap){
      parLoopDataSS  << opMap->getType().getAsString() << " "
                 << opMap->getName().str() << " "
                 << opMap->getDefinition(opDat->getASTContext())
                        ->getLocation()
                        .printToString(*SM);
    } else {
      parLoopDataSS << "OP_ID";
    }
    parLoopDataSS << "\n";

    parLoopDataSS << "  dim:" << getIntValFromExpr(argCallExpr->getArg(3)->IgnoreCasts())
               << "\n";
    parLoopDataSS << "  type:" 
               << getAsStringLiteral(argCallExpr->getArg(4))->getString().str()
               << "\n";
    parLoopDataSS << "  access:" << getIntValFromExpr(argCallExpr->getArg(5)->IgnoreCasts()) //TODO map back values.
               << "\n";
  }

  llvm::outs() << parLoopDataSS.str();
}

void ParLoopHandler::run(const matchers::MatchFinder::MatchResult &Result) {
  // TODO collect data about kernel.
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
  clang::SourceLocation sLoc = function->getLocStart();
  clang::FileID fileID = sourceManager->getFileID(sLoc);
  const clang::FileEntry *fileEntry = sourceManager->getFileEntryForID(fileID);
  const std::string fname = fileEntry->getName();

  clang::tooling::Replacements &Rpls = (*Replace)[fname];
  clang::tooling::Replacement Rep(*sourceManager, parent->getLocStart(), 0,
                                  func_signature);

  (*Replace)[fname] = Rpls.merge(clang::tooling::Replacements(Rep));
    // Add replacement for func call
    // TODO try with SourceRange instead of length
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

}
