#ifndef OMP4DIREXTUSERFUNCREFTOOL
#define OMP4DIREXTUSERFUNCREFTOOL
#include "core/OP2WriteableRefactoringTool.hpp"
#include "core/OPParLoopData.h"
#include "generators/common/handler.hpp"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
using namespace clang::ast_matchers;

template <bool OMP4 = false>
class VecUserFuncHandler
    : public clang::ast_matchers::MatchFinder::MatchCallback {

  const ParLoop &loop;
  const std::vector<size_t> &redIndexes;
  std::map<std::string, clang::tooling::Replacements> *Replace;
  std::map<std::string, std::vector<clang::SourceLocation>> alreadyDone;

  int addLocalVarDecls(const MatchFinder::MatchResult &Result) {
    const clang::FunctionDecl *match =
        Result.Nodes.getNodeAs<clang::FunctionDecl>("userFuncDecl");
    if (!match)
      return 1;
    clang::SourceManager *sm = Result.SourceManager;
    // FIXME filename always /tmp/loop.h
    std::string filename = sm->getFilename(match->getBeginLoc());
    std::string localdefs, addlocals;
    for (const size_t &i : redIndexes) {
      const OPArg &arg = loop.getArg(i);
      std::string name = loop.getUserFuncInfo().paramNames[i];
      std::string type = arg.type;

      if (arg.accs == OP_accs_type::OP_INC) {
        localdefs += type + " __" + name + "_l = 0;";
        addlocals += "*" + name + "+= " + "__" + name + "_l;";
      }
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
        match->getBody()->getBeginLoc().getLocWithOffset(1), 0, localdefs);
    if (llvm::Error err = (*Replace)[filename].add(repl2)) {
      // TODO diagnostics..
      llvm::errs() << "Replacement for key: "
                   << "userFuncDecl"
                   << " (reduction adding in the end) "
                   << " failed in: " << filename << "\n";
    }
    if (OMP4) {
      SourceRange replRange(
          match->getBeginLoc().getLocWithOffset(
              ("inline void " + loop.getUserFuncInfo().funcName).length()),
          match->getBody()->getBeginLoc().getLocWithOffset(-1));
      std::string funcSignature = "_vec(";
      llvm::raw_string_ostream os(funcSignature);
      for (size_t i = 0; i < loop.getNumArgs(); ++i) {
        const OPArg &arg = loop.getArg(i);
        if (arg.accs == OP_accs_type::OP_READ) {
          os << "const ";
        }
        os << arg.type << " ";
        if (loop.getArg(i).isDirect() || loop.getArg(i).isGBL) {
          os << "*" << loop.getUserFuncInfo().paramNames[i] << ",";
        } else {
          os << loop.getUserFuncInfo().paramNames[i] << "[*][SIMD_VEC],";
        }
      }
      os << " int idx)";
      os.str();
      tooling::Replacement funcNameRepl(*sm, CharSourceRange(replRange, false),
                                        funcSignature);
      if (llvm::Error err = (*Replace)[filename].add(funcNameRepl)) {
        // TODO diagnostics..
        llvm::errs() << "Replacement for key: "
                     << "userFuncDecl"
                     << " (_vec signature) "
                     << " failed in: " << filename << "\n";
      }
    }
    return 0;
  }

  int handleIndirects(const MatchFinder::MatchResult &Result, std::string key) {
    const clang::ArraySubscriptExpr *match =
        Result.Nodes.getNodeAs<clang::ArraySubscriptExpr>(key);
    if (!match)
      return 1;
    clang::SourceManager *sm = Result.SourceManager;
    // FIXME filename always /tmp/loop.h
    clang::SourceLocation end =
        sm->getSpellingLoc(match->getEndLoc()).getLocWithOffset(1);
    for (const clang::SourceLocation &sl :
         alreadyDone[key]) { // TODO find better solution
      if (end == sl) {
        return 0;
      }
    }
    alreadyDone[key].push_back(end);
    std::string filename =
        sm->getFilename(sm->getSpellingLoc(match->getBeginLoc()));

    tooling::Replacement repl(*Result.SourceManager, end, 0, "[idx]");
    if (llvm::Error err = (*Replace)[filename].add(repl)) {
      // TODO diagnostics..
      llvm::errs() << "Replacement for key: " << key
                   << " failed in: " << filename << "\n";
    }
    return 0;
  }

public:
  VecUserFuncHandler(
      const ParLoop &_loop, const std::vector<size_t> &redIndexes,
      std::map<std::string, clang::tooling::Replacements> *Replace)
      : loop(_loop), redIndexes(redIndexes), Replace(Replace) {}
  virtual void run(const MatchFinder::MatchResult &Result) override {

    for (const size_t &i : redIndexes) {
      std::string name = loop.getUserFuncInfo().paramNames[i];
      if (!fixLengthReplHandler<clang::DeclRefExpr, -1>(
              Result, Replace, name + "_red", name.size() + 1,
              [name]() { return "__" + name + "_l"; }))
        return;
    }
    if (OMP4 && !loop.isDirect()) {
      for (size_t i = 0; i < loop.getNumArgs(); ++i) {
        if (!loop.getArg(i).isDirect() && !loop.getArg(i).isGBL) {
          std::string name = loop.getUserFuncInfo().paramNames[i];
          if (!handleIndirects(Result, name + "_indirect"))
            return;
          if (!fixLengthReplHandler<clang::DeclRefExpr, -1>(
                  Result, Replace, name + "_indirect", name.size() + 1,
                  [name]() { return name + "[0][idx]"; }))
            return;
        }
      }
    }
    if (!addLocalVarDecls(Result))
      return;
  }
};

class VecUserFuncGenerator : public OP2WriteableRefactoringTool {
  const ParLoop &loop;
  const std::vector<size_t> &redIndexes;

public:
  VecUserFuncGenerator(const clang::tooling::CompilationDatabase &Compilations,
                       const ParLoop &_loop,
                       const std::vector<size_t> &redIndexes,
                       std::vector<std::string> path = {"/tmp/loop.h"})
      : OP2WriteableRefactoringTool(Compilations, path), loop(_loop),
        redIndexes(redIndexes) {}

  template <bool OMP4 = false> std::string run() {

    clang::ast_matchers::MatchFinder Finder;
    VecUserFuncHandler<OMP4> handler(loop, redIndexes, &getReplacements());

    Finder.addMatcher(functionDecl(isDefinition(),
                                   hasName(loop.getUserFuncInfo().funcName),
                                   parameterCountIs(loop.getNumArgs()))
                          .bind("userFuncDecl"),
                      &handler);
    for (const size_t &i : redIndexes) {
      if (loop.getArg(i).accs == OP_accs_type::OP_INC) {
        std::string name = loop.getUserFuncInfo().paramNames[i];
        Finder.addMatcher(declRefExpr(to(parmVarDecl(hasName(name))),
                                      hasAncestor(functionDecl(hasName(
                                          loop.getUserFuncInfo().funcName))))
                              .bind(name + "_red"),
                          &handler);
      }
    }
    if (OMP4 && !loop.isDirect()) {
      for (size_t i = 0; i < loop.getNumArgs(); ++i) {
        if (!loop.getArg(i).isDirect() && !loop.getArg(i).isGBL) {
          std::string name = loop.getUserFuncInfo().paramNames[i];
          Finder.addMatcher(
              arraySubscriptExpr(hasBase(hasDescendant(declRefExpr(
                                     to(parmVarDecl(hasName(name))),
                                     hasAncestor(functionDecl(hasName(
                                         loop.getUserFuncInfo().funcName)))))))
                  .bind(name + "_indirect"),
              &handler);
          Finder.addMatcher(
              unaryOperator(hasOperatorName("*"),
                            hasUnaryOperand(hasDescendant(
                                declRefExpr(to(parmVarDecl(hasName(name))))
                                    .bind(name + "_indirect"))),
                            hasAncestor(functionDecl(
                                hasName(loop.getUserFuncInfo().funcName)))),
              &handler);
        }
      }
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
