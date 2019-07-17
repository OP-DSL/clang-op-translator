#ifndef USERFUNCTRANSFORMATOR_H
#define USERFUNCTRANSFORMATOR_H
#include "core/OP2WriteableRefactoringTool.hpp"
#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "generators/common/handler.hpp"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace OP2 {
using namespace clang::ast_matchers;

class SoaTransformationHandler
    : public clang::ast_matchers::MatchFinder::MatchCallback {
  const ParLoop &loop;
  std::map<std::string, clang::tooling::Replacements> *Replace;

public:
  SoaTransformationHandler(
      const ParLoop &_loop,
      std::map<std::string, clang::tooling::Replacements> *Replace)
      : loop(_loop), Replace(Replace) {}
  virtual void run(const MatchFinder::MatchResult &Result) override {

    for (size_t i = 0; i < loop.getNumArgs(); ++i) {
      const OPArg &arg = loop.getArg(i);
      if (arg.dim > 1 && ((arg.isDirect() && !arg.isGBL) || !arg.isDirect())) {
        std::string name = loop.getUserFuncInfo().paramNames[i];
        std::string key = name + "_accs";

        const clang::ArraySubscriptExpr *match =
            Result.Nodes.getNodeAs<clang::ArraySubscriptExpr>(key);
        if (!match)
          continue;
        clang::SourceManager *sm = Result.SourceManager;
        clang::SourceLocation end = sm->getSpellingLoc(match->getEndLoc());
        std::string filename =
            sm->getFilename(sm->getSpellingLoc(match->getBeginLoc()));
        clang::SourceLocation idxBegin =
            sm->getSpellingLoc(match->getIdx()->getBeginLoc());

        tooling::Replacement repl(*Result.SourceManager, idxBegin, 0, "(");
        if (llvm::Error err = (*Replace)[filename].add(repl)) {
          // TODO diagnostics..
          llvm::errs() << "Replacement for key: " << key
                       << " failed in: " << filename << "\n";
        }
        std::string strideName =
            (arg.isDirect() ? "direct"
                            : ("opDat" + std::to_string(loop.dataIdxs[i]))) +
            "_" + loop.getName() + "_stride_OP2CONSTANT";
        tooling::Replacement repl2(*Result.SourceManager, end, 0,
                                   ")*" + strideName);
        if (llvm::Error err = (*Replace)[filename].add(repl2)) {
          // TODO diagnostics..
          llvm::errs() << "Replacement for key: " << key
                       << " failed in: " << filename << "\n";
        }
      }
    }
  }
};

class UserFuncTransformator : public OP2WriteableRefactoringTool {
  const ParLoop &loop;
  OP2Optimizations op2Flags;

public:
  UserFuncTransformator(const clang::tooling::CompilationDatabase &Compilations,
                        const ParLoop &_loop, const OP2Optimizations &flags,
                        std::vector<std::string> path = {"/tmp/loop.cu"})
      : OP2WriteableRefactoringTool(Compilations, path), loop(_loop),
        op2Flags(flags) {}

  std::string run() {

    clang::ast_matchers::MatchFinder Finder;
    SoaTransformationHandler handler(loop, &getReplacements());

    // Add mavhers to handler
    for (size_t i = 0; i < loop.getNumArgs(); ++i) {
      const OPArg &arg = loop.getArg(i);
      // Only need strides for nonglobal data with dim > 1 and don't need stride
      // for indirect increments with 2 level coloring
      if (arg.dim > 1 && ((arg.isDirect() && !arg.isGBL) ||
                          (!arg.isDirect() && (op2Flags.staging != OP_STAGE_ALL ||
                                               arg.accs != OP_INC)))) {
        std::string name = loop.getUserFuncInfo().paramNames[i];
        Finder.addMatcher(
            arraySubscriptExpr(hasBase(hasDescendant(declRefExpr(
                                   to(parmVarDecl(hasName(name))),
                                   hasAncestor(functionDecl(hasName(
                                       loop.getUserFuncInfo().funcName)))))))
                .bind(name + "_accs"),
            &handler);
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

#endif /* USERFUNCTRANSFORMATOR_H */
