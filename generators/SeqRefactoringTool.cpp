#include "SeqRefactoringTool.h"
#include "BaseKernelHandler.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
//-------------------
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
//-------------------

namespace OP2 {

SeqRefactoringTool::SeqRefactoringTool(
    const clang::tooling::CompilationDatabase &Compilations,
    const ParLoop &loop,
    std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps)
    : RefactoringTool(Compilations, {skeletons[loop.getKernelType()]},
                      PCHContainerOps),
      loop(loop) {}

int SeqRefactoringTool::generateKernelFile() {
  using namespace clang::ast_matchers;
  llvm::outs() << "Started\n";

  // Create Callbacks
  BaseKernelHandler baseKernelHandler(&getReplacements(), loop);
  clang::ast_matchers::MatchFinder Finder;
  Finder.addMatcher(BaseKernelHandler::parLoopDeclMatcher, &baseKernelHandler);

  if (int Result =
          run(clang::tooling::newFrontendActionFactory(&Finder).get())) {
    llvm::outs() << "Error " << Result << "\n";
    return Result;
  }

  // TODO modify everithing under this line:
  // OP2::writeOutReplacements(Tool);
  // Set up the Rewriter (For this we need a SourceManager)
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
      new clang::DiagnosticOptions();
  clang::DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
          new clang::DiagnosticIDs()),
      &*DiagOpts, new clang::TextDiagnosticPrinter(llvm::errs(), &*DiagOpts),
      true);
  clang::SourceManager Sources(Diagnostics, getFiles());

  // Apply all replacements to a rewriter.
  clang::Rewriter Rewrite(Sources, clang::LangOptions());
  applyAllReplacements(Rewrite);

  // Query the rewriter for all the files it has rewritten, dumping their new
  // contents to stdout.
  for (clang::Rewriter::buffer_iterator I = Rewrite.buffer_begin(),
                                        E = Rewrite.buffer_end();
       I != E; ++I) {
    const clang::FileEntry *Entry = Sources.getFileEntryForID(I->first);
    llvm::outs() << "Rewrite buffer for file: " << Entry->getName() << "\n";

    std::error_code ec;
    std::string filename = loop.getName() + "_seqkernel.cpp";

    llvm::outs() << filename << "\n";
    llvm::raw_fd_ostream outfile{llvm::StringRef(filename), ec,
                                 llvm::sys::fs::F_Text | llvm::sys::fs::F_RW};
    I->second.write(outfile);
  }
  return 0;
}

} // namespace OP2
