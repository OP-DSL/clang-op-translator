#ifndef OPDATAREGISTER_HPP
#define OPDATAREGISTER_HPP
#include "OPParLoopData.h"
#include "utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace OP2 {
namespace matchers = clang::ast_matchers;

class DataRegister : public matchers::MatchFinder::MatchCallback {
  std::map<std::string, const op_set> &sets;
  std::map<std::string, const op_map> &mappings;
  std::set<op_global_const> &gbls;

public:
  DataRegister(std::map<std::string, const op_set> &sets,
               std::map<std::string, const op_map> &mappings,
               std::set<op_global_const> &constants)
      : sets(sets), mappings(mappings), gbls(constants) {}

  void addToFinder(matchers::MatchFinder &finder) {
    using namespace matchers;
    finder.addMatcher(callExpr(callee(functionDecl(hasName("op_decl_const"))))
                          .bind("op_decl_const"),
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
    }
  }
};

} // namespace OP2
#endif /* ifndef OPDATAREGISTER_HPP */
