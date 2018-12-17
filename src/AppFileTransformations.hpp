#ifndef APPLICATIONFILETRANSFORMATIONS_HPP
#define APPLICATIONFILETRANSFORMATIONS_HPP
#include "core/OPParLoopData.h"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Refactoring.h>

namespace op_dsl {
namespace matchers = clang::ast_matchers;

class AppFileRefactoringTool;

/// @brief SourceFileCallbacks for replace op_seq.h and ops_seq.h includes.
///
/// Replace #include "op_seq.h" to op_lib_cpp.h include and op_par_loop,
/// and #include "ops_seq.h" to ops_lib_cpp.h include and ops_par_loop
/// declarations.
class ParLoopDeclarator : public clang::tooling::SourceFileCallbacks {

private:
  class IncludeFinderPPCallback : public clang::PPCallbacks {
    clang::CompilerInstance *CI;
    ParLoopDeclarator *callback;

  public:
    IncludeFinderPPCallback(clang::CompilerInstance *CI,
                            ParLoopDeclarator *callback);

    virtual void InclusionDirective(
        clang::SourceLocation HashLoc, const clang::Token &,
        clang::StringRef FileName, bool, clang::CharSourceRange FilenameRange,
        const clang::FileEntry *File, clang::StringRef, clang::StringRef,
        const clang::Module *, clang::SrcMgr::CharacteristicKind) override;
  };

public:
  explicit ParLoopDeclarator(AppFileRefactoringTool &tool);
  virtual bool handleBeginSource(clang::CompilerInstance &CI) override;
  virtual void handleEndSource() override;
  void addFunction(const std::string &funcDeclaration);
  void setCurrentFile(std::string, clang::SourceRange, clang::SourceManager *,
                      const std::string &);

private:
  /// @brief The text that will be inserted in place of op_seq.h include. Starts
  /// with #include "op_lib_cpp.h". Reseted in every handleEndSource() call.
  std::string functionDeclarations{""};
  /// @brief SourceRange of the #include "op_seq.h" or #include "ops_seq.h"  to
  /// be replaced.
  clang::SourceRange replRange;
  /// @brief The filename for the current MainFile
  std::string fileName{""};
  AppFileRefactoringTool &tool;
  clang::SourceManager *SM = nullptr;
};

/**
 * @brief Utility to replace op and ops par_loop calls.
 *
 * Expects a match on an op_par_loop or an ops_par_loop call with "par_loop"
 * key. Effect: Replace op_par_loop(kernel,...) call with a call to the
 * generated par_loop function: op_par_loop_kernel(...). Works similarly for ops
 * loops. Also notifies the \c ParLoopDeclarator about the par_loops.
 */
class ParloopCallReplaceOperation {
public:
  ParloopCallReplaceOperation(OPApplication &, ParLoopDeclarator &,
                              AppFileRefactoringTool &);

  void operator()(const matchers::MatchFinder::MatchResult &Result) const;

private:
  OPApplication &app; /**< The application which stores the par_loops */
  ParLoopDeclarator &parLoopDeclarator;
  AppFileRefactoringTool &tool;
};

/**
 * @brief Utility to replace op and ops decl_const calls.
 *
 * Expects a match on an op_decl_const or an ops_decl_const call with
 * "decl_const" key. Replace these calls with op_decl_const2 and ops_decl_const2
 * calls.
 */
class OPConstDeclarationReplaceOperation {
public:
  explicit OPConstDeclarationReplaceOperation(AppFileRefactoringTool &);
  void operator()(const matchers::MatchFinder::MatchResult &Result) const;

private:
  AppFileRefactoringTool &tool;
};

} // namespace op_dsl
#endif /* ifndef APPLICATIONFILETRANSFORMATIONS_HPP */
