#ifndef OPDATAREGISTER_HPP
#define OPDATAREGISTER_HPP
#include "OPParLoopData.h"
#include "utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

class DataRegister : public matchers::MatchFinder::MatchCallback {
  std::map<std::string, std::string> &sets;
  std::map<std::string, const op_map> &mappings;
  std::set<op_global_const> &gbls;

public:
  DataRegister(std::map<std::string, std::string> &sets,
               std::map<std::string, const op_map> &mappings,
               std::set<op_global_const> &constants)
      : sets(sets), mappings(mappings), gbls(constants) {}

  void addToFinder(matchers::MatchFinder &finder) {
    using namespace matchers;
    finder.addMatcher(callExpr(callee(functionDecl(hasName("op_decl_const"))))
                          .bind("op_decl_const"),
                      this);
    finder.addMatcher(
        varDecl(hasInitializer(
                    callExpr(callee(functionDecl(hasName("op_decl_set"))))
                        .bind("op_decl_set")))
            .bind("op_set"),
        this);
    finder.addMatcher(
        varDecl(hasInitializer(
                    callExpr(callee(functionDecl(hasName("op_decl_set_hdf5"))))
                        .bind("op_decl_set")))
            .bind("op_set"),
        this);
    finder.addMatcher(callExpr(callee(functionDecl(hasName("op_decl_map"))))
                          .bind("op_decl_map"),
                      this);
    finder.addMatcher(
        callExpr(callee(functionDecl(hasName("op_decl_map_hdf5"))))
            .bind("op_decl_map"),
        this);
  }

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override {
    const clang::CallExpr *callExpr =
        Result.Nodes.getNodeAs<clang::CallExpr>("op_decl_const");
    if (callExpr) {
      unsigned size = getIntValFromExpr(callExpr->getArg(0));
      std::string type =
          getAsStringLiteral(callExpr->getArg(1))->getString().str();
      std::string name =
          getExprAsDecl<clang::VarDecl>(callExpr->getArg(2))->getNameAsString();
      op_global_const c(type, name, size);
      size_t oldSize = gbls.size();
      gbls.insert(c);
      if (gbls.size() == oldSize) {
        llvm::errs() << "Multiple global consts defined with same name:" << c
                     << " vs " << *gbls.find(c) << "\n";
      }
      llvm::outs() << c << "\n";
    } else if ((callExpr =
                    Result.Nodes.getNodeAs<clang::CallExpr>("op_decl_set"))) {
      const clang::VarDecl *var =
          Result.Nodes.getNodeAs<clang::VarDecl>("op_set");
      std::string name = getAsStringLiteral(callExpr->getArg(1))->getString();
      std::string varname = var->getNameAsString();
      auto it = sets.find(name);
      if (sets.end() != it) {
        llvm::errs() << "Multiple sets defined with same name:" << name << ": "
                     << varname << "vs" << it->second << "\n";
      }
      sets[name] = varname;
      llvm::outs() << name << "\n";
    } else if ((callExpr =
                    Result.Nodes.getNodeAs<clang::CallExpr>("op_decl_map"))) {
      std::string name = getAsStringLiteral(callExpr->getArg(4))->getString();
      auto it = mappings.find(name);
      unsigned dim = getIntValFromExpr(callExpr->getArg(2));
      std::string varnameFrom =
          getExprAsDecl<clang::VarDecl>(callExpr->getArg(0))->getNameAsString();
      std::string varnameTo =
          getExprAsDecl<clang::VarDecl>(callExpr->getArg(1))->getNameAsString();
      std::string from, to;
      for (auto nameTOvarname : sets) {
        if (nameTOvarname.second == varnameFrom) {
          from = nameTOvarname.first;
        }
        if (nameTOvarname.second == varnameTo) {
          to = nameTOvarname.first;
        }
      }
      const op_map m(from, to, dim, name);
      if (it != mappings.end()) {
        llvm::errs() << "Multiple mappings defined with same name:" << m
                     << " vs " << it->second << "\n";
      }
      mappings.insert(std::pair<std::string, const op_map>(name, m));
      llvm::outs() << m << "\n";
    }
  }
};

} // namespace OP2
#endif /* ifndef OPDATAREGISTER_HPP */
