#ifndef ASTMATCHERSEXTENSION_OP2
#define ASTMATCHERSEXTENSION_OP2
#include "clang/AST/StmtOpenMP.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
/// \brief Matches omp parallel for directives.
///
/// Example matches omp parallel for
/// \code
///   #pragma omp parallel for
///   for(;;){}
/// \endcode
const clang::ast_matchers::internal::VariadicDynCastAllOfMatcher<
    clang::Stmt, clang::OMPParallelForDirective>
    ompParallelForDirective;
#endif /* ifndef ASTMATCHERSEXTENSION_OP2 */
