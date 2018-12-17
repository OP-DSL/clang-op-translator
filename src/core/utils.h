#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <cassert>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Refactoring.h>

namespace op_dsl {

template <unsigned N>
clang::DiagnosticBuilder reportDiagnostic(
    const clang::ASTContext &Context, const clang::Expr *expr,
    const char (&FormatString)[N],
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Error,
    clang::SourceLocation *sl = nullptr) {
  clang::DiagnosticsEngine &DiagEngine = Context.getDiagnostics();
  auto DiagID = DiagEngine.getCustomDiagID(level, FormatString);
  auto SourceRange = expr->getSourceRange();
  auto report = DiagEngine.Report(sl ? *sl : SourceRange.getBegin(), DiagID);
  report.AddSourceRange({SourceRange, true});
  return report;
}

inline std::string getSourceAsString(const clang::SourceRange d,
                                     const clang::SourceManager *sm) {
  clang::SourceLocation b(d.getBegin());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(
      d.getEnd(), 0, *sm, clang::LangOptions()));
  return std::string(sm->getCharacterData(b),
                     sm->getCharacterData(e) - sm->getCharacterData(b));
}

inline std::string decl2str(const clang::Decl *d,
                            const clang::SourceManager *sm) {
  return getSourceAsString(d->getSourceRange(), sm);
}

inline std::vector<std::string>
getCommandlineArgs(clang::tooling::CommonOptionsParser &parser) {
  clang::tooling::CompilationDatabase &c = parser.getCompilations();
  clang::tooling::CompileCommand myC = c.getCompileCommands("")[0];
  std::vector<std::string> ToolCommandLine;
  std::copy(&myC.CommandLine[1], &myC.CommandLine[myC.CommandLine.size() - 1],
            std::back_inserter(ToolCommandLine));
  return ToolCommandLine;
}

inline void tryToEvaluateICE(
    int &value, const clang::Expr *probablyICE,
    const clang::ASTContext &Context, const std::string &what,
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Error,
    const std::string &extra = "") {
  llvm::APSInt APvalue;
  clang::SourceLocation sl = probablyICE->getBeginLoc();
  if (!probablyICE->isIntegerConstantExpr(APvalue, Context, &sl, false)) {
    reportDiagnostic(Context, probablyICE, "%0 is not a constant expression %1",
                     level, &sl)
        << what << extra;
    return;
  }
  value = APvalue.getExtValue();
}

template <typename T> inline const T *getExprAsDecl(const clang::Expr *expr) {
  if (const auto *declRefExpr =
          llvm::dyn_cast<clang::DeclRefExpr>(expr->IgnoreCasts())) {
    const auto *decl = llvm::dyn_cast<T>(declRefExpr->getFoundDecl());
    assert(decl);
    return decl;
  }
  if (const auto *declRefExpr = llvm::dyn_cast<clang::DeclRefExpr>(
          expr->child_begin()->IgnoreImplicit())) {
    const auto *decl = llvm::dyn_cast<T>(declRefExpr->getFoundDecl());
    assert(decl);
    return decl;
  }
  expr->dumpColor();
  assert(false && "Failed to get Decl from Expr.");
  return nullptr;
}

inline const clang::StringLiteral *getAsStringLiteral(const clang::Expr *expr) {
  if (auto str = llvm::dyn_cast<clang::StringLiteral>(expr))
    return str;

  auto cast = llvm::dyn_cast<clang::CastExpr>(expr);
  if (nullptr == cast)
    return nullptr;
  return llvm::dyn_cast<clang::StringLiteral>(cast->getSubExpr());
}

inline bool isStringLiteral(const clang::Expr &expr) {
  return getAsStringLiteral(&expr) != nullptr;
}

inline llvm::raw_ostream &debugs() {
#ifndef NDEBUG
  return llvm::errs();
#else
  return llvm::nulls();
#endif
}

template <typename F>
class MatchMaker : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  F matchFunction;

public:
  MatchMaker() : matchFunction() {}
  template <typename... Param>
  explicit MatchMaker(Param... param) : matchFunction(param...) {}
  explicit MatchMaker(F f) : matchFunction{f} {}
  virtual ~MatchMaker() = default;
  void
  run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override {
    matchFunction(Result);
  }
};

template <typename F> MatchMaker<F> make_matcher(F matchFunction) {
  return MatchMaker<F>{matchFunction};
}

template <typename T>
const T *findParent(const clang::Stmt &stmt, clang::ASTContext &context) {
  auto vec = context.getParents(stmt);

  if (vec.empty())
    return nullptr;

  if (const T *t = vec[0].get<T>()) {
    return t;
  }

  const auto *pStmt = vec[0].get<clang::Stmt>();
  if (pStmt) {
    return findParent<T>(*pStmt, context);
  }

  return nullptr;
}

template <typename FileNameGenerator>
inline void writeReplacementsTo(const FileNameGenerator &generator,
                                clang::tooling::RefactoringTool *tool) {
  // Set up the Rewriter (For this we need a SourceManager)
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
      new clang::DiagnosticOptions();
  clang::DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
          new clang::DiagnosticIDs()),
      &*DiagOpts, new clang::TextDiagnosticPrinter(llvm::errs(), &*DiagOpts),
      true);
  clang::SourceManager Sources(Diagnostics, tool->getFiles());

  // Apply all replacements to a rewriter.
  clang::Rewriter Rewrite(Sources, clang::LangOptions());
  formatAndApplyAllReplacements(tool->getReplacements(), Rewrite, "LLVM");

  // Query the rewriter for all the files it has rewritten, dumping their new
  // contents to output files.
  for (clang::Rewriter::buffer_iterator I = Rewrite.buffer_begin(),
                                        E = Rewrite.buffer_end();
       I != E; ++I) {
    std::string filename = generator(Sources.getFileEntryForID(I->first));
    std::error_code ec;
    llvm::raw_fd_ostream outfile{llvm::StringRef(filename), ec,
                                 llvm::sys::fs::F_Text};
    I->second.write(outfile);
  }
}

} // namespace op_dsl
#endif // end of header guard
