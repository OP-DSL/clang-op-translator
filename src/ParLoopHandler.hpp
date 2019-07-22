#ifndef PARLOOPHANDLER_H_INCLUDED
#define PARLOOPHANDLER_H_INCLUDED
#include "core/OPParLoopData.h"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Refactoring.h>

namespace OP2 {
class OP2RefactoringTool;

namespace matchers = clang::ast_matchers;

/// @brief SourceFileCallbacks for replace op_seq.h includes.
///
/// Replace #include "op_seq.h" to op_lib_cpp.h include and op_par_loop
/// declarations
class OPParLoopDeclarator : public clang::tooling::SourceFileCallbacks {

private:
  class IncludeFinderPPCallback : public clang::PPCallbacks {
    clang::CompilerInstance *CI;
    OPParLoopDeclarator *callback;

  public:
    IncludeFinderPPCallback(clang::CompilerInstance *CI,
                            OPParLoopDeclarator *callback);

    virtual void
    InclusionDirective(clang::SourceLocation HashLoc,
                       const clang::Token &includeTok, clang::StringRef FileName, bool,
                       clang::CharSourceRange FilenameRange,
                       const clang::FileEntry *File, clang::StringRef SearchPath,
                       clang::StringRef RelativePath, const clang::Module *Imported,
                       clang::SrcMgr::CharacteristicKind);
  };

public:
  OPParLoopDeclarator(OP2RefactoringTool &tool);
  virtual bool handleBeginSource(clang::CompilerInstance &CI) override;
  virtual void handleEndSource() override;
  void addFunction(std::string funcDeclaration);
  void setCurrentFile(std::string, clang::SourceRange, clang::SourceManager *);

private:
  /// @brief The text that will be inserted in place of op_seq.h include. Starts
  /// with #include "op_lib_cpp.h". Reseted in every handleEndSource() call.
  std::string functionDeclarations;
  OP2RefactoringTool &tool;
  /// @brief SourceRange of the #include "op_seq.h" to be replaced.
  clang::SourceRange replRange;
  /// @brief The filename for the current MainFile
  std::string fileName;
  clang::SourceManager *SM;
};

class ParLoopHandler : public matchers::MatchFinder::MatchCallback {
  OP2RefactoringTool &tool;
  OP2Application &app;
  OPParLoopDeclarator &declarator;

  void parseFunctionDecl(const clang::CallExpr *parloopExpr,
                         const clang::SourceManager *SM);

public:
  ParLoopHandler(OP2RefactoringTool &tool, OP2Application &app,
                 OPParLoopDeclarator &declarator);

  virtual void run(const matchers::MatchFinder::MatchResult &Result) override;
};
} // namespace OP2
#endif // end of PARLOOPHANDLER_H_INCLUDED guard
