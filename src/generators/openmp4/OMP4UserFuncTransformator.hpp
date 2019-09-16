#ifndef USERFUNCTRANSFORMATOR_H
#define USERFUNCTRANSFORMATOR_H
#include "core/OP2WriteableRefactoringTool.hpp"
#include "core/OPParLoopData.h"
#include "core/op2_clang_core.h"
#include "generators/common/handler.hpp"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"
#include <iostream>

namespace OP2 {
using namespace clang::ast_matchers;

class OMP4TransformationHandler
    : public clang::ast_matchers::MatchFinder::MatchCallback {
  const ParLoop &loop;
  std::map<std::string, clang::tooling::Replacements> *Replace;
  const OP2Application &application;
  std::vector <std::string> &const_list;
  std::vector <std::string> &kernel_arg_name;

public:
  OMP4TransformationHandler(
      const ParLoop &_loop,
      std::map<std::string, clang::tooling::Replacements> *Replace, const OP2Application &application, std::vector <std::string> &const_list, std::vector <std::string> &kernel_arg_name)
      : loop(_loop), Replace(Replace), application(application), const_list(const_list), kernel_arg_name(kernel_arg_name) {}
  virtual void run(const MatchFinder::MatchResult &Result) override {

    for (auto it=application.constants.begin(); it != application.constants.end(); ++it) {

        std::string key = it->name + "_accs";

        const clang::DeclRefExpr *match =
            Result.Nodes.getNodeAs<clang::DeclRefExpr>(key);
        if (!match)
          continue;

      	std::vector<std::string>::iterator itr_list = find (const_list.begin(), const_list.end(), it->name);
      	if(itr_list == const_list.end())
     		const_list.push_back(it->name);
    }

    // getting function argument names
	std::string key = loop.getUserFuncInfo().funcName + "_accs";
    const clang::FunctionDecl *match =
        Result.Nodes.getNodeAs<clang::FunctionDecl>(key);
    if (!match)
      return;

  	for (int i = 0; i < match->getNumParams(); i++){
        kernel_arg_name.push_back(getSourceAsString(match->getParamDecl(i)->getSourceRange(), Result.SourceManager));
  	}
  }
};

class OMP4UserFuncTransformator : public OP2WriteableRefactoringTool {
  const ParLoop &loop;
  const OP2Application &application;
  std::vector <std::string> &const_list;
  std::vector <std::string> &kernel_arg_name;
  /*OP2Optimizations op2Flags;*/

public:
  OMP4UserFuncTransformator(const clang::tooling::CompilationDatabase &Compilations,
                        const ParLoop &_loop,  const OP2Application &application, std::vector <std::string> &const_list,
                        std::vector <std::string> &kernel_arg_name, std::vector<std::string> path = {"/tmp/loop.cu"})
      : OP2WriteableRefactoringTool(Compilations, path), loop(_loop), application(application), const_list(const_list), kernel_arg_name(kernel_arg_name) {}


  
  std::string run() {

    clang::ast_matchers::MatchFinder Finder;
    OMP4TransformationHandler handler(loop, &getReplacements(), application, const_list, kernel_arg_name);
    for (auto it=application.constants.begin(); it != application.constants.end(); ++it){
      Finder.addMatcher(
            	declRefExpr(to(varDecl(hasName(it->name))), 
                                   hasAncestor(functionDecl(hasName(
                                       loop.getUserFuncInfo().funcName))))
                .bind(it->name + "_accs"), &handler);
    }

    // matcher to function argument
    Finder.addMatcher(functionDecl(hasName(loop.getUserFuncInfo().funcName))
                .bind(loop.getUserFuncInfo().funcName + "_accs"), &handler);

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
