#pragma once
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace OP2 {

const clang::StringLiteral *getAsStringLiteral(const clang::Expr *expr) {
  if (auto str = llvm::dyn_cast<clang::StringLiteral>(expr))
    return str;

  auto cast = llvm::dyn_cast<clang::CastExpr>(expr);
  if (!cast)
    return nullptr;
  return llvm::dyn_cast<clang::StringLiteral>(cast->getSubExpr());
}

bool isStringLiteral(const clang::Expr &expr) {
  return getAsStringLiteral(&expr);
}

template <unsigned N>
clang::DiagnosticBuilder reportDiagnostic(
    const clang::ASTContext &Context, const clang::Expr *expr,
    const char (&FormatString)[N],
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Error) {
  clang::DiagnosticsEngine &DiagEngine = Context.getDiagnostics();
  auto DiagID = DiagEngine.getCustomDiagID(level, FormatString);
  auto SourceRange = expr->getSourceRange();
  auto report = DiagEngine.Report(SourceRange.getBegin(), DiagID);
  report.AddSourceRange({SourceRange, true});
  return report;
}

llvm::raw_ostream& debugs() {
#ifndef NDEBUG
    return llvm::errs();
#else
    return llvm::nulls();
#endif

}

template <typename F> class MatchMaker : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  F matchFunction;
public:
  MatchMaker(F f) : matchFunction{f} {}
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override {
    matchFunction(Result);
  }
};

template <typename F>
MatchMaker<F> make_matcher(F matchFunction) {
  return MatchMaker<F>{matchFunction};
}

template <typename T>
const T *findParent(const clang::Stmt &stmt,
                                  clang::ASTContext &context) {
  auto vec = context.getParents(stmt);

  if (vec.empty())
    return nullptr;

  if (const T *t = vec[0].get<T>()) {
    return t;
  }

  const clang::Stmt *pStmt = vec[0].get<clang::Stmt>();
  if (pStmt) {
    return findParent<T>(*pStmt, context);
  }

  return nullptr;
}
}
