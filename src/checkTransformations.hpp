#ifndef CHECKTRANSFORMATIONS_HPP
#define CHECKTRANSFORMATIONS_HPP
#include "core/OPParLoopData.h"
#include "core/utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <vector>

namespace OP2 {

/**
 * @brief Utility to check wheter a function called once.
 *
 * Can be used with \c OP2::MatchMaker, counts the number of calls to a function
 * and gives warning for multiple function calls. The destructor checks if the
 * function is called at all. Usage:
 * \code
 *   OP2::MatchMaker<CheckSingleCallOperation> m("op_init");
 *   finder.addMatcher(
 *     callExpr(callee(functionDecl(hasName("op_init")))).bind("callExpr"),
 *     &m);
 * \endcode
 */
class CheckSingleCallOperation {
  std::vector<std::string>
      Locations; /**< The SourceLocations of the function calls.*/
  std::string functionName;

public:
  CheckSingleCallOperation(const CheckSingleCallOperation &) = delete;
  CheckSingleCallOperation(std::string name) : functionName(name) {}

  void operator()(const clang::ast_matchers::MatchFinder::MatchResult &Result) {
    using namespace clang;
    if (const CallExpr *initCall =
            Result.Nodes.getNodeAs<CallExpr>("callExpr")) {
      const SourceManager *sm = Result.SourceManager;
      SourceLocation sl = initCall->getBeginLoc();
      if (Locations.size()) {
        DiagnosticsEngine &DE = sm->getDiagnostics();
        auto diagBuilder = DE.Report(
            sl, DE.getCustomDiagID(
                    DiagnosticsEngine::Warning,
                    "Multiple call for %0 function found. First call: %1."));
        diagBuilder.AddString(initCall->getDirectCallee()->getName());
        diagBuilder.AddString(Locations[0]);
        diagBuilder.AddSourceRange({initCall->getSourceRange(), true});
      }
      Locations.push_back(sl.printToString(*sm));
    }
  }

  ~CheckSingleCallOperation() {
    if (Locations.size() == 0) {
      llvm::errs() << "Warning: No " << functionName << " call found.\n";
    }
  }
};

class ParLoopParser {
public:
  ParLoopParser(OPApplication &_app) : application(_app){};

  OPArg
  parseOPArg(const clang::Expr *argExpr, const int idx,
             const clang::ast_matchers::MatchFinder::MatchResult &Result) {
    const clang::Stmt *argStmt = llvm::dyn_cast<clang::Stmt>(argExpr);
    // ugly solution to get the op_arg_** callExpr from from AST..
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
    std::string fname =
        llvm::dyn_cast<clang::NamedDecl>(argCallExpr->getCalleeDecl())
            ->getNameAsString();
    bool isGBL = fname == "op_arg_gbl";
    assert((isGBL || fname == "op_arg_dat") && "Unknown arg declaration.");
    std::string opDat = getSourceAsString(
        argCallExpr->getArg(0)->getSourceRange(), Result.SourceManager);
    size_t argIdx = 3 - (isGBL ? 2 : 0);
    const clang::Expr *probablyICE =
        argCallExpr->getArg(argIdx++)->IgnoreCasts();
    clang::SourceLocation sl = probablyICE->getBeginLoc();
    auto result = tryToEvaluateICE(probablyICE, *Result.Context, sl);
    size_t dim = 0;
    if (!result) {
      reportDiagnostic(*Result.Context, probablyICE,
                       "Dimension of op_arg is not a constant expression",
                       clang::DiagnosticsEngine::Warning, &sl);
    } else {
      dim = *result;
    }
    std::string type =
        getAsStringLiteral(argCallExpr->getArg(argIdx++))->getString().str();
    OP_accs_type accs = OP2::OP_RW;
    probablyICE = argCallExpr->getArg(argIdx)->IgnoreCasts();
    result = tryToEvaluateICE(probablyICE, *Result.Context, sl);
    if (!result) {
      reportDiagnostic(
          *Result.Context, probablyICE,
          "Access descriptor of an op_arg is not constant expression",
          clang::DiagnosticsEngine::Error, &sl);
    } else {
      accs = OP_accs_type(*result);
    }
    if (isGBL) {
      return OPArg(idx, opDat, dim, type, accs);
    }

    int mapidx = -2;
    std::string opMap = "";
    probablyICE = argCallExpr->getArg(1)->IgnoreCasts();
    result = tryToEvaluateICE(probablyICE, *Result.Context, sl);
    if (!result) {
      reportDiagnostic(
          *Result.Context, probablyICE,
          "Mapping index is not a constant expression inside an op_arg",
          clang::DiagnosticsEngine::Error, &sl);
    } else {
      mapidx = *result;
    }
    if (mapidx != -1 && result) {
      opMap = getExprAsDecl<clang::VarDecl>(argCallExpr->getArg(2))
                  ->getNameAsString();
    }

    return OPArg(idx, opDat, mapidx, opMap, dim, type, accs);
  }

  UserFuncData
  parseUserFunc(const clang::FunctionDecl *funcD,
                const clang::ast_matchers::MatchFinder::MatchResult &Result) {

    std::string path =
        funcD->getBeginLoc().printToString(*Result.SourceManager);
    path = path.substr(0, path.find(":"));
    std::vector<std::string> paramNames;
    for (size_t i = 0; i < funcD->getNumParams(); ++i) {
      paramNames.push_back(funcD->getParamDecl(i)->getNameAsString());
    }

    return UserFuncData(decl2str(funcD, Result.SourceManager),
                        funcD->getNameAsString(), funcD->isInlineSpecified(),
                        path, paramNames);
  }

  void operator()(const clang::ast_matchers::MatchFinder::MatchResult &Result) {
    // inital checks and daignostics
    const clang::CallExpr *parLoopCall =
        Result.Nodes.getNodeAs<clang::CallExpr>("par_loop");
    if (parLoopCall->getNumArgs() < 3) {
      reportDiagnostic(*Result.Context, parLoopCall,
                       "not enough arguments to op_par_loop");
    }
    const clang::Expr *str_arg = parLoopCall->getArg(1);
    const clang::FunctionDecl *fDecl =
        getExprAsDecl<clang::FunctionDecl>(parLoopCall->getArg(0));

    if (!fDecl) {
      reportDiagnostic(*Result.Context, parLoopCall->getArg(0),
                       "Must be a function pointer");
      return;
    } else if (!fDecl->hasBody()) {
      reportDiagnostic(
          *Result.Context, parLoopCall->getArg(0),
          "body must be available at the point of an op_par_loop call");
    }
    // end of basic diagnostics. Parsing op_args:
    std::vector<OPArg> args;
    for (unsigned arg_ind = 3; arg_ind < parLoopCall->getNumArgs(); ++arg_ind) {
      args.push_back(parseOPArg(parLoopCall->getArg(arg_ind), arg_ind, Result));
    }
    // end of argument parsing. Parse user function:
    UserFuncData userFuncData = parseUserFunc(fDecl, Result);
    // end of user function parsing. Add loop to application.
    application.addParLoop(ParLoop(userFuncData, userFuncData.funcName, args));
  }

private:
  OPApplication &application;
};

} // namespace OP2
#endif /* ifndef CHECKTRANSFORMATIONS_HPP */
