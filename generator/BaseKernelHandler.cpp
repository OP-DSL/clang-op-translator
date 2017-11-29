#include "BaseKernelHandler.h"
#include "../utils.h"
#include "handler.hpp"

namespace {
using namespace clang::ast_matchers;
const DeclarationMatcher parLoopSkeletonMatcher =
    functionDecl(hasName("op_par_loop_skeleton"));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
///__________________________________MATCHERS__________________________________
// Static Matchers of BaseKernelHandler
const DeclarationMatcher BaseKernelHandler::parLoopDeclMatcher =
    functionDecl(hasName("op_par_loop_skeleton")).bind("par_loop_decl");
const DeclarationMatcher BaseKernelHandler::nargsMatcher =
    varDecl(hasType(isInteger()), hasName("nargs"),
            hasAncestor(parLoopSkeletonMatcher))
        .bind("nargs_decl");

const DeclarationMatcher BaseKernelHandler::argsArrMatcher =
    varDecl(hasName("args"), hasAncestor(parLoopSkeletonMatcher))
        .bind("args_arr_decl");
const StatementMatcher BaseKernelHandler::argsArrSetterMatcher =
    cxxOperatorCallExpr(/*hasType(cxxRecordDecl(hasName("op_arg"))),FIXME more
                           specific matcher*/
                        hasAncestor(parLoopSkeletonMatcher))
        .bind("args_element_setter");
const StatementMatcher BaseKernelHandler::opTimingReallocMatcher =
    callExpr(callee(functionDecl(hasName("op_timing_realloc"))),
             hasAncestor(parLoopSkeletonMatcher))
        .bind("op_timing_realloc");
const StatementMatcher BaseKernelHandler::printfKernelNameMatcher =
    callExpr(callee(functionDecl(hasName("printf"))), hasAncestor(ifStmt()),
             hasAncestor(parLoopSkeletonMatcher))
        .bind("printfName");
const StatementMatcher BaseKernelHandler::opKernelsSubscriptMatcher =
    arraySubscriptExpr(hasBase(implicitCastExpr(hasSourceExpression(
                           declRefExpr(to(varDecl(hasName("OP_kernels"))))))),
                       hasIndex(integerLiteral(equals(0))),
                       hasParent(memberExpr(hasParent(binaryOperator().bind(
                                                "op_kernels_assignment")))
                                     .bind("opk_member_expr")),
                       hasAncestor(parLoopSkeletonMatcher))
        .bind("op_kernels_index");
const DeclarationMatcher BaseKernelHandler::nindsMatcher =
    varDecl(hasType(isInteger()), hasName("ninds"),
            hasAncestor(parLoopSkeletonMatcher))
        .bind("ninds_decl");

const DeclarationMatcher BaseKernelHandler::indsArrMatcher =
    varDecl(hasName("inds"), hasAncestor(parLoopSkeletonMatcher))
        .bind("inds_arr_decl");
const StatementMatcher BaseKernelHandler::opMPIReduceMatcher =
    callExpr(
        callee(functionDecl(hasName("op_mpi_reduce"), parameterCountIs(2))),
        hasAncestor(parLoopSkeletonMatcher))
        .bind("reduce_func_call");
const StatementMatcher BaseKernelHandler::opMPIWaitAllIfStmtMatcher =
    ifStmt(hasThen(compoundStmt(statementCountIs(1),
                                hasAnySubstatement(callExpr(callee(functionDecl(
                                    hasName("op_mpi_wait_all"))))))))
        .bind("wait_all_if");

///________________________________CONSTRUCTORS________________________________
BaseKernelHandler::BaseKernelHandler(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const ParLoop &loop)
    : Replace(Replace), loop(loop) {}

///_______________________________GLOBAL_HANDLER_______________________________
void BaseKernelHandler::run(const MatchFinder::MatchResult &Result) {
  if (!handleParLoopDecl(Result))
    return; // if successfully handled return
  if (!lineReplHandler<clang::VarDecl, 1>(
          Result, Replace, "nargs_decl", [this]() {
            return "int nargs = " + std::to_string(this->loop.getNumArgs());
          })) //handleNargsDecl
    return;
  if (!lineReplHandler<clang::VarDecl>(
          Result, Replace, "args_arr_decl", [this]() {
            return "op_arg args[" + std::to_string(this->loop.getNumArgs());
          })) // handleArgsArrDecl
    return;
  if (!lineReplHandler<clang::CXXOperatorCallExpr, 5>(
          Result, Replace, "args_element_setter",
          std::bind(&BaseKernelHandler::handleArgsArrSetter,
                    this))) // handleArgsArrSetter
    return;
  if (!HANDLER(clang::CallExpr, 3, "op_timing_realloc",
               BaseKernelHandler::handleOPTimingRealloc))
    return;
  if (!HANDLER(clang::CallExpr, 3, "printfName",
               BaseKernelHandler::handleOPDiagPrintf))
    return;
  if (!handleOPKernels(Result))
    return;
  if (!lineReplHandler<clang::VarDecl, 1>(
          Result, Replace, "ninds_decl", [this]() {
            std::set<std::string> inds;
            for (unsigned i = 0; i < this->loop.getNumArgs(); ++i) {
              if (!this->loop.getArg(i).isDirect()) {
                inds.insert(this->loop.getArg(i).opDat);
              }
            }
            return "int ninds = " + std::to_string(inds.size());
          })) //handleNindsDecl
    return;
  if (!HANDLER(clang::VarDecl, 3, "inds_arr_decl",
               BaseKernelHandler::handleIndsArr)) // handleIndsArrDecl
    return;
  if (!lineReplHandler<CallExpr, 2>(
          Result, Replace, "reduce_func_call",
          [this]() { return this->loop.getMPIReduceCall(); }))
    return; // if successfully handled return
  if (!handleMPIWaitAllIfStmt(Result))
    return; // if successfully handled return
}

///__________________________________HANDLERS__________________________________
int BaseKernelHandler::handleParLoopDecl(
    const MatchFinder::MatchResult &Result) {
  const clang::FunctionDecl *function =
      Result.Nodes.getNodeAs<clang::FunctionDecl>("par_loop_decl");
  if (!function)
    return 1; // We shouldn't handle this match
  clang::SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(function->getLocStart(), sm);
  // replace skeleton to the name of the loop
  size_t nameoffset = std::string("void op_par_loop_").length();
  size_t length = std::string("skeleton").length();
  clang::tooling::Replacement funcNameRep(
      *sm, function->getLocStart().getLocWithOffset(nameoffset), length,
      loop.getName());
  if (llvm::Error err = (*Replace)[filename].add(funcNameRep)) {
    // TODO diagnostics..
    llvm::errs() << "Function name replacement failed in: " << filename << "\n";
  }
  // add op_args to the parameter list
  std::string arg_repl = "";
  llvm::raw_string_ostream os(arg_repl);
  for (size_t i = 1; i < loop.getNumArgs(); ++i) {
    os << ", op_arg arg" << i;
  }

  if (loop.getNumArgs() > 1) {
    clang::tooling::Replacement funcArgRep(
        *sm,
        function->getParamDecl(function->getNumParams() - 1)
            ->getLocEnd()
            .getLocWithOffset(4 /*FIXME hardcoded length of arg0*/),
        0, os.str());
    if (llvm::Error err = (*Replace)[filename].add(funcArgRep)) {
      // TODO diagnostics..
      llvm::errs() << "Function arg addition failed in: " << filename << "\n";
    }
  }
  return 0;
}

std::string BaseKernelHandler::handleArgsArrSetter() {
  std::string replacement = "";
  llvm::raw_string_ostream os(replacement);
  for (unsigned i = 0; i < loop.getNumArgs(); ++i) {
    os << "args[" << i << "] = arg" << i << ";\n";
  }
  return os.str();
}

std::string BaseKernelHandler::handleIndsArr() {
  std::string indarr =
      "int inds[" + std::to_string(this->loop.getNumArgs()) + "] = {";
  llvm::raw_string_ostream os(indarr);

  std::map<std::string, int> datToInd;
  for (unsigned i = 0; i < loop.getNumArgs(); ++i) {
    if (loop.getArg(i).isDirect()) {
      os << "-1, ";
    } else {
      auto it = datToInd.find(loop.getArg(i).opDat);
      if (it != datToInd.end()) {
        os << it->second << ", ";
      } else {
        os << datToInd.size() << ", ";
        datToInd[loop.getArg(i).opDat] = datToInd.size() - 1;
      }
    }
  }
  os.str();

  return indarr.substr(0, indarr.size() - 2) + "};\n";
}

std::string BaseKernelHandler::handleOPTimingRealloc() {
  return "op_timing_realloc(" + std::to_string(loop.getLoopID()) + ");";
}

std::string BaseKernelHandler::handleOPDiagPrintf() {
  std::string replString = std::string("\" kernel routine ") +
                           (loop.isDirect() ? "w/o" : "with") +
                           " indirection:  " + loop.getName() + "\"";
  return "printf(" + replString + ");";
}

int BaseKernelHandler::handleOPKernels(const MatchFinder::MatchResult &Result) {
  const clang::ArraySubscriptExpr *kernelsSubscriptExpr =
      Result.Nodes.getNodeAs<clang::ArraySubscriptExpr>("op_kernels_index");
  if (!kernelsSubscriptExpr)
    return 1; // We shouldn't handle this match

  clang::SourceManager *sm = Result.SourceManager;
  std::string filename =
      getFileNameFromSourceLoc(kernelsSubscriptExpr->getLocStart(), sm);

  if (Result.Nodes.getNodeAs<clang::MemberExpr>("opk_member_expr")
          ->getMemberDecl()
          ->getNameAsString() == "transfer") {
    const clang::BinaryOperator *bop =
        Result.Nodes.getNodeAs<clang::BinaryOperator>("op_kernels_assignment");
    if (getIntValFromExpr(bop->getRHS()->IgnoreImpCasts()) == 0) {
      clang::SourceRange replRange(
          bop->getLocStart(),
          bop->getLocEnd().getLocWithOffset(3)); // TODO proper end
      // clang::arcmt::trans::findSemiAfterLocation(bop->getLocEnd(),
      // *Result.Context));
      clang::tooling::Replacement repl(*sm,
                                       clang::CharSourceRange(replRange, false),
                                       loop.getTransferData());
      if (llvm::Error err = (*Replace)[filename].add(repl)) {
        // TODO diagnostics..
        llvm::errs() << "Set transfer failed in: " << filename << "\n";
      }
      return 0;
    }
  }
  clang::tooling::Replacement repl(
      *sm, kernelsSubscriptExpr->getIdx()->getLocStart(), 1,
      std::to_string(loop.getLoopID()));
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Set looID in for index OP_kernels failed in: " << filename
                 << "\n";
  }

  return 0;
}

int BaseKernelHandler::handleMPIWaitAllIfStmt(
    const MatchFinder::MatchResult &Result) {
  const IfStmt *ifStmt = Result.Nodes.getNodeAs<IfStmt>("wait_all_if");
  if (!ifStmt)
    return 1;
  if (!loop.isDirect())
    return 0;

  SourceManager *sm = Result.SourceManager;
  std::string filename = getFileNameFromSourceLoc(ifStmt->getLocStart(), sm);
  SourceRange replRange(ifStmt->getLocStart(),
                        ifStmt->getLocEnd().getLocWithOffset(1));
  /*FIXME magic number for semicolon pos*/

  tooling::Replacement repl(*sm, CharSourceRange(replRange, false), "");

  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement of op_mpi_wat_all failed in: " << filename
                 << "\n";
  }
  return 0;
}

} // end of namespace OP2
