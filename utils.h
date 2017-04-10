#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"

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
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Warning) {
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

}
