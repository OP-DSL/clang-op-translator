#ifndef APPLICATIONFILETRANSFORMATIONS_HPP
#define APPLICATIONFILETRANSFORMATIONS_HPP
#include "core/OPParLoopData.h"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Refactoring.h>

namespace OP2 {
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

    virtual void InclusionDirective(clang::SourceLocation HashLoc,
                                    const clang::Token &, StringRef FileName,
                                    bool, clang::CharSourceRange FilenameRange,
                                    const clang::FileEntry *File, StringRef,
                                    StringRef, const clang::Module *,
                                    clang::SrcMgr::CharacteristicKind) override;
  };

public:
  ParLoopDeclarator(AppFileRefactoringTool &tool);
  virtual bool handleBeginSource(clang::CompilerInstance &CI) override;
  virtual void handleEndSource() override;
  void addFunction(std::string funcDeclaration);
  void setCurrentFile(std::string, clang::SourceRange, clang::SourceManager *,
                      std::string);

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

class ParloopCallReplaceOperation {
public:
  ParloopCallReplaceOperation(OPApplication &, ParLoopDeclarator &,
                              AppFileRefactoringTool &);
  void operator()(const matchers::MatchFinder::MatchResult &Result) const;

private:
  OPApplication &app;
  ParLoopDeclarator &parLoopDeclarator;
  AppFileRefactoringTool &tool;
};

} // namespace OP2
#endif /* ifndef APPLICATIONFILETRANSFORMATIONS_HPP */
