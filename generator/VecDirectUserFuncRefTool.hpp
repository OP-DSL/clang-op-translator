#ifndef VECDIREXTUSERFUNCREFTOOL
#define VECDIREXTUSERFUNCREFTOOL
#include "../OP2WriteableRefactoringTool.hpp"
#include "../OPParLoopData.h"
#include "handler.hpp"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
using namespace clang::ast_matchers;

class VecDirUserFuncHandler
    : public clang::ast_matchers::MatchFinder::MatchCallback {

  const ParLoop &loop;
  const std::vector<size_t> &redIndexes;
  std::map<std::string, clang::tooling::Replacements> *Replace;

  int addLocalVarDecls(const MatchFinder::MatchResult &Result) {
    const clang::FunctionDecl *match =
        Result.Nodes.getNodeAs<clang::FunctionDecl>("userFuncDecl");
    if (!match)
      return 1;
    llvm::outs() << "userFuncDecl"
                 << "\n";
    clang::SourceManager *sm = Result.SourceManager;
    std::string filename = getFileNameFromSourceLoc(match->getLocStart(), sm);
    std::string localdefs, addlocals;
    for (const size_t &i : redIndexes) {
      std::string name = loop.getUserFuncInfo().paramNames[i];
      std::string type = loop.getArg(i).type;
      localdefs += type + " __" + name + "_l;";
      addlocals += "*" + name + "+= " + "__" + name + "_l;";
    }
    clang::SourceLocation end = match->getBodyRBrace();

    tooling::Replacement repl(*Result.SourceManager, end.getLocWithOffset(-1),
                              0, addlocals);
    if (llvm::Error err = (*Replace)[filename].add(repl)) {
      // TODO diagnostics..
      llvm::errs() << "Replacement for key: "
                   << "userFuncDecl"
                   << " (reduction adding in the end) "
                   << " failed in: " << filename << "\n";
    }
    tooling::Replacement repl2(
        *Result.SourceManager,
        match->getBody()->getLocStart().getLocWithOffset(1), 0, localdefs);
    if (llvm::Error err = (*Replace)[filename].add(repl2)) {
      // TODO diagnostics..
      llvm::errs() << "Replacement for key: "
                   << "userFuncDecl"
                   << " (reduction adding in the end) "
                   << " failed in: " << filename << "\n";
    }

    return 0;
  }

public:
  VecDirUserFuncHandler(
      const ParLoop &_loop, const std::vector<size_t> &redIndexes,
      std::map<std::string, clang::tooling::Replacements> *Replace)
      : loop(_loop), redIndexes(redIndexes), Replace(Replace) {}
  virtual void run(const MatchFinder::MatchResult &Result) override {

    for (const size_t &i : redIndexes) {
      std::string name = loop.getUserFuncInfo().paramNames[i];
      if (!fixLengthReplHandler<clang::DeclRefExpr, -1, true>(
              Result, Replace, name + "_red", name.size() + 1,
              [name]() { return "__" + name + "_l"; }))
        return;
      if (!addLocalVarDecls(Result))
        return;
    }
  }
};

class VecDirectUserFuncGenerator : public OP2WriteableRefactoringTool {
  const ParLoop &loop;
  const std::vector<size_t> &redIndexes;

public:
  VecDirectUserFuncGenerator(
      const clang::tooling::CompilationDatabase &Compilations,
      const ParLoop &_loop, const std::vector<size_t> &redIndexes)
      : OP2WriteableRefactoringTool(Compilations,
                                    {_loop.getUserFuncInfo().path}),
        loop(_loop), redIndexes(redIndexes) {}

  std::string run() {

    clang::ast_matchers::MatchFinder Finder;
    VecDirUserFuncHandler handler(loop, redIndexes, &getReplacements());

    Finder.addMatcher(functionDecl(isDefinition(),
                                   hasName(loop.getUserFuncInfo().funcName),
                                   parameterCountIs(loop.getNumArgs()))
                          .bind("userFuncDecl"),
                      &handler);
    for (const size_t &i : redIndexes) {
      std::string name = loop.getUserFuncInfo().paramNames[i];
      Finder.addMatcher(declRefExpr(to(parmVarDecl(hasName(name))),
                                    hasAncestor(functionDecl(hasName(
                                        loop.getUserFuncInfo().funcName))))
                            .bind(name + "_red"),
                        &handler);
    }

    // run the tool
    if (int Result = OP2WriteableRefactoringTool::run(
            clang::tooling::newFrontendActionFactory(&Finder).get())) {
      llvm::outs() << "Error " << Result << "\n";
    }
    std::string funcDecl;
    llvm::raw_string_ostream os(funcDecl);
    writeReplacementsTo(os);
    return os.str();
  }

  virtual std::string
  getOutputFileName(const clang::FileEntry *f = nullptr) const {
    return "";
  }
};

} // namespace OP2

#endif /* ifndef VECDIREXTUSERFUNCREFTOOL */
