#include "ParLoopHandler.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include <memory>
#include <sstream>
#include "op_par_loop.h"

static llvm::cl::OptionCategory Op2Category("OP2 Options");
static llvm::cl::extrahelp
CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

namespace OP2 {
//Subclass Refactoringtool (or ClangTool? RefactoringTool has a map for Replacements)
class OP2RefactoringTool : public clang::tooling::RefactoringTool {
  // We can collect all data about kernels
  std::vector<ParLoop> loops;
public:
  OP2RefactoringTool(
      const clang::tooling::CompilationDatabase &Compilations,
      llvm::ArrayRef<std::string> SourcePaths,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : clang::tooling::RefactoringTool(Compilations, SourcePaths,
                                        PCHContainerOps) {}
  std::vector<ParLoop>& getParLoops(){
    return loops;
  }
};

//write the modifications to output files
void writeOutReplacements(clang::tooling::RefactoringTool &Tool) {
  // Set up the Rewriter (For this we need a SourceManager)
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
      new clang::DiagnosticOptions();
  clang::DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
          new clang::DiagnosticIDs()),
      &*DiagOpts, new clang::TextDiagnosticPrinter(llvm::errs(), &*DiagOpts),
      true);
  clang::SourceManager Sources(Diagnostics, Tool.getFiles());

  // Apply all replacements to a rewriter.
  clang::Rewriter Rewrite(Sources, clang::LangOptions());
  Tool.applyAllReplacements(Rewrite);

  // Query the rewriter for all the files it has rewritten, dumping their new
  // contents to stdout.
  for (clang::Rewriter::buffer_iterator I = Rewrite.buffer_begin(),
                                        E = Rewrite.buffer_end();
       I != E; ++I) {
    const clang::FileEntry *Entry = Sources.getFileEntryForID(I->first);
    llvm::outs() << "Rewrite buffer for file: " << Entry->getName() << "\n";

    std::error_code ec;
    std::string filename = Entry->getName().str();
    size_t basename_start = filename.rfind("/") + 1,
           basename_end = filename.rfind(".");
    if (basename_start == std::string::npos)
      basename_start = 0;
    if (basename_end == std::string::npos || basename_end < basename_start)
      llvm::errs() << "Invalid filename: " << Entry->getName() << "\n";
    filename = filename.substr(basename_start, basename_end - basename_start) +
               "_op.cpp";
    llvm::raw_fd_ostream outfile{llvm::StringRef(filename), ec,
                                 llvm::sys::fs::F_Text | llvm::sys::fs::F_RW};

    I->second.write(outfile);
  }
}

} // namespace OP2

int main(int argc, const char **argv) {
  using namespace clang::tooling;
  using namespace clang::ast_matchers;
  CommonOptionsParser OptionsParser(argc, argv, Op2Category);

  OP2::OP2RefactoringTool Tool(OptionsParser.getCompilations(),
                              OptionsParser.getSourcePathList());

  OP2::ParLoopHandler parLoopHandlerCallback(&Tool.getReplacements(), 
                                             Tool.getParLoops());
 
  clang::ast_matchers::MatchFinder Finder;
  Finder.addMatcher(
      callExpr(callee(functionDecl(hasName("op_par_loop")))).bind("par_loop"),
      &parLoopHandlerCallback);

  if (int Result = Tool.run(newFrontendActionFactory(&Finder).get())) {
    return Result;
  }

  OP2::writeOutReplacements(Tool);

  return 0;
}
