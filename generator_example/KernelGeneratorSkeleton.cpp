#include "KernelHandlerSkeleton.hpp"
#include "generators/common/handler.hpp"

namespace {
using namespace clang::ast_matchers;
const auto parLoopSkeletonCompStmtMatcher =
    compoundStmt(hasParent(functionDecl(hasName("op_par_loop_skeleton"))));
} // namespace

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;
///__________________________________MATCHERS__________________________________

const DeclarationMatcher KernelHandlerSkeleton::ExampleMatcher =
    varDecl().bind("Example_key");

///________________________________CONSTRUCTORS________________________________
KernelHandlerSkeleton::KernelHandlerSkeleton(
    std::map<std::string, clang::tooling::Replacements> *Replace,
    const OP2Application &app, size_t loopIdx)
    : Replace(Replace), app(app), loopIdx(loopIdx) {}
///_______________________________GLOBAL_HANDLER_______________________________
void KernelHandlerSkeleton::run(const MatchFinder::MatchResult &Result) {

  if (!lineReplHandler<VarDecl>(Result, Replace, "Example_key",
                                []() { return ""; }))
    return; // if successfully handled return
}
///__________________________________HANDLERS__________________________________
} // namespace OP2
