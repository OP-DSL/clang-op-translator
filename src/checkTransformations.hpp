#ifndef CHECKTRANSFORMATIONS_HPP
#define CHECKTRANSFORMATIONS_HPP
#include "core/OPParLoopData.h"
#include "core/utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <vector>

namespace op_dsl {

/**
 * @brief Utility to check wheter a function called once.
 *
 * Can be used with \c OP2::MatchMaker, counts the number of calls to a function
 * and gives warning for multiple function calls. The destructor checks if the
 * function is called at all. Check for key "callExpr" inside the MatchResult.
 * Usage:
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
  CheckSingleCallOperation(const CheckSingleCallOperation &&) = delete;
  CheckSingleCallOperation &
  operator=(const CheckSingleCallOperation &) = delete;
  CheckSingleCallOperation &
  operator=(const CheckSingleCallOperation &&) = delete;

  explicit CheckSingleCallOperation(std::string name)
      : functionName(std::move(name)) {}

  void operator()(const clang::ast_matchers::MatchFinder::MatchResult &Result) {
    if (const auto *initCall =
            Result.Nodes.getNodeAs<clang::CallExpr>("callExpr")) {
      const clang::SourceManager *sm = Result.SourceManager;
      clang::SourceLocation sl = initCall->getBeginLoc();
      if (!Locations.empty()) {
        clang::DiagnosticsEngine &DE = sm->getDiagnostics();
        auto diagBuilder = DE.Report(
            sl, DE.getCustomDiagID(
                    clang::DiagnosticsEngine::Warning,
                    "Multiple call for %0 function found. First call: %1."));
        diagBuilder.AddString(initCall->getDirectCallee()->getName());
        diagBuilder.AddString(Locations[0]);
        diagBuilder.AddSourceRange({initCall->getSourceRange(), true});
      }
      Locations.push_back(sl.printToString(*sm));
    }
  }

  ~CheckSingleCallOperation() {
    if (Locations.empty()) {
      llvm::errs() << "Warning: No " << functionName << " call found.\n";
    }
  }
};

/**
 * @brief Utility to parse par_loops.
 *
 * Takes a reference of an \c OPApplication and fills it with the data about
 * loops. Expect a match on a CallExpr with "par_loop" key.
 */
class ParLoopParser {

  /**
   * @brief Parses argument of par_loops.
   */
  OPArg
  parseOPArg(const clang::Expr *argExpr, const int idx,
             const clang::ast_matchers::MatchFinder::MatchResult &Result) {
    const auto *argStmt = llvm::dyn_cast<clang::Stmt>(argExpr);
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
    const auto *argCallExpr = llvm::dyn_cast<clang::CallExpr>(argStmt);
    std::string fname =
        llvm::dyn_cast<clang::NamedDecl>(argCallExpr->getCalleeDecl())
            ->getNameAsString();
    // argument kind
    OPArg::OPArgKind kind = OPArg::OP_DAT;
    // get indices for arguments
    int dat_Idx = 0, mapIdx_Idx = -1, mapOrStencil_Idx = -1, dim_Idx = 1,
        type_Idx = argCallExpr->getNumArgs() - 2,
        accs_Idx = argCallExpr->getNumArgs() - 1;
    bool optional = false;
    if (fname == "op_arg_gbl" || fname == "ops_arg_gbl") {
      kind = OPArg::OP_GBL;
    } else if (fname == "ops_arg_reduce") {
      kind = OPArg::OP_REDUCE;
    } else if (fname == "op_arg_dat") {
      mapIdx_Idx = 1;
      mapOrStencil_Idx = 2;
      dim_Idx = 3;
    } else if (fname == "ops_arg_dat") {
      mapOrStencil_Idx = 2;
    } else if (fname == "op_opt_arg_dat") {
      optional = true;
      dat_Idx = 1;
      mapIdx_Idx = 2;
      mapOrStencil_Idx = 3;
      dim_Idx = 4;
    } else if (fname == "ops_arg_dat_opt") {
      mapOrStencil_Idx = 2;
      optional = true;
      accs_Idx--;
      type_Idx--;
    } else if (fname == "ops_arg_idx") {
      return OPArg(idx);
    } else {
      assert(false && "Unknown arg declaration.");
    }
    // get varname for op_dat/ops_dat/reduction_handler
    std::string opDat = getSourceAsString(
        argCallExpr->getArg(dat_Idx)->getSourceRange(), Result.SourceManager);
    // get dimension of the data
    int dim = 0;
    tryToEvaluateICE(dim, argCallExpr->getArg(dim_Idx)->IgnoreCasts(),
                     *Result.Context, "dimension of op_arg",
                     clang::DiagnosticsEngine::Warning,
                     "this may prevent further optimizations");
    // get type information
    std::string type =
        getAsStringLiteral(argCallExpr->getArg(type_Idx))->getString().str();
    // get access descriptor
    int temp = 0;
    tryToEvaluateICE(temp, argCallExpr->getArg(accs_Idx)->IgnoreCasts(),
                     *Result.Context, "access descriptor of op_arg");
    auto accs = OP_accs_type(temp);

    if (kind == OPArg::OP_GBL || kind == OPArg::OP_REDUCE) {
      if (accs == OP_INC || accs == OP_MAX || accs == OP_MIN) {
        kind = OPArg::OP_REDUCE;
      } else if (kind == OPArg::OP_REDUCE) {
        clang::SourceLocation sl = argCallExpr->getArg(accs_Idx)->getBeginLoc();
        reportDiagnostic(*Result.Context, argCallExpr,
                         "reduction variables must be accessed with OP_MAX, "
                         "OP_MIN or OP_INC",
                         clang::DiagnosticsEngine::Error, &sl);
      }
      return OPArg(idx, opDat, dim, type, accs, kind);
    }
    // get mapping in case of op_dat
    int mapidx = -2;
    if (mapIdx_Idx != -1) {
      tryToEvaluateICE(mapidx, argCallExpr->getArg(mapIdx_Idx)->IgnoreCasts(),
                       *Result.Context, "mapping index of op_arg");
    }
    // get op_map or ops_stencil varname
    std::string opMap;
    if (mapidx != -1 && mapOrStencil_Idx != -1) {
      opMap =
          getExprAsDecl<clang::VarDecl>(argCallExpr->getArg(mapOrStencil_Idx))
              ->getNameAsString();
    }

    return OPArg(idx, opDat, dim, type, accs, kind, mapidx, opMap, optional);
  }

  UserFuncData
  parseUserFunc(const clang::FunctionDecl *funcD,
                const clang::ast_matchers::MatchFinder::MatchResult &Result) {

    std::string path =
        funcD->getBeginLoc().printToString(*Result.SourceManager);
    path = path.substr(0, path.find(':'));
    std::vector<std::string> paramNames;
    for (size_t i = 0; i < funcD->getNumParams(); ++i) {
      paramNames.push_back(funcD->getParamDecl(i)->getNameAsString());
    }

    return UserFuncData(decl2str(funcD, Result.SourceManager),
                        funcD->getNameAsString(), funcD->isInlineSpecified(),
                        paramNames);
  }

public:
  explicit ParLoopParser(OPApplication &_app) : application(_app){};

  void operator()(const clang::ast_matchers::MatchFinder::MatchResult &Result) {
    // inital checks and daignostics
    const auto *parLoopCall =
        Result.Nodes.getNodeAs<clang::CallExpr>("par_loop");
    DSL kind = parLoopCall->getDirectCallee()->getName() == "ops_par_loop"
                   ? DSL::OPS
                   : DSL::OP2;
    if (parLoopCall->getNumArgs() < 3 ||
        (kind == OPS && parLoopCall->getNumArgs() < 5)) {
      reportDiagnostic(*Result.Context, parLoopCall,
                       "not enough arguments to op_par_loop");
    }
    const auto *fDecl =
        getExprAsDecl<clang::FunctionDecl>(parLoopCall->getArg(0));

    if (nullptr == fDecl) {
      reportDiagnostic(*Result.Context, parLoopCall->getArg(0),
                       "must be a function pointer");
      return;
    }
    if (!fDecl->hasBody()) {
      reportDiagnostic(
          *Result.Context, parLoopCall->getArg(0),
          "body must be available at the point of an op_par_loop call");
    }
    // end of basic diagnostics. Parsing op_args:
    std::vector<OPArg> args;
    for (unsigned arg_ind = kind == OPS ? 5 : 3;
         arg_ind < parLoopCall->getNumArgs(); ++arg_ind) {
      args.push_back(parseOPArg(parLoopCall->getArg(arg_ind), arg_ind, Result));
    }
    // end of argument parsing. Parse user function:
    UserFuncData userFuncData = parseUserFunc(fDecl, Result);
    // end of user function parsing. Add loop to application.
    application.addParLoop(ParLoop(userFuncData, userFuncData.funcName, args));
  }

private:
  OPApplication
      &application; /**< The application where we collect the data about loops*/
};

/**
 * @brief Utility to search for op2 and ops constants and register them.
 *
 * Search for op_decl_const and ops_decl_const calls to parse them and add the
 * constants to the application modell. Expect match for the "decl_const" key on
 * a CallExpr of an op_decl_const or an ops_decl_const function.
 *
 */
class OPConstRegister {
public:
  explicit OPConstRegister(OPApplication &app) : application(app) {}

  void operator()(const clang::ast_matchers::MatchFinder::MatchResult &Result) {
    const auto *callExpr =
        Result.Nodes.getNodeAs<clang::CallExpr>("decl_const");
    // ops_decl_const has one more argument (the first is name)
    int arg_offset = static_cast<int>(callExpr->getDirectCallee()->getName() ==
                                      "ops_decl_const");

    int dim = 0;
    // TODO(bgd54): dimension is constexpr?
    tryToEvaluateICE(dim, callExpr->getArg(0 + arg_offset)->IgnoreCasts(),
                     *Result.Context,
                     "dimension of global constant in a decl_const call");
    std::string type =
        getAsStringLiteral(callExpr->getArg(1 + arg_offset))->getString().str();
    std::string name =
        getExprAsDecl<clang::VarDecl>(callExpr->getArg(2 + arg_offset))
            ->getNameAsString();
    op_global_const c(type, name, dim);
    size_t oldSize = application.constants.size();
    application.constants.insert(c);
    if (application.constants.size() == oldSize) {
      reportDiagnostic(*Result.Context, callExpr,
                       "redefinition of global consts with name %0")
          << name;
    }
  }

private:
  OPApplication &application; /**< The application where we collect the data
                                 about constants*/
};

} // namespace op_dsl
#endif /* ifndef CHECKTRANSFORMATIONS_HPP */
