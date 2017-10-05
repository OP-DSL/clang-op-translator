#ifndef OP2REFACTORINGTOOL_HPP
#define OP2REFACTORINGTOOL_HPP 
// clang Tooling includes
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

//OP2 includes
#include "ParLoopHandler.h"
#include "generators/SeqRefactoringTool.h"
//Parloops.

namespace OP2 {

class OP2RefactoringTool : public clang::tooling::RefactoringTool {
protected:
  clang::tooling::CommonOptionsParser &optionsParser;
  // We can collect all data about kernels
  std::vector<ParLoop> loops;

public:
  OP2RefactoringTool(
      clang::tooling::CommonOptionsParser &optionsParser,
      std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps =
          std::make_shared<clang::PCHContainerOperations>())
      : clang::tooling::RefactoringTool(optionsParser.getCompilations(),
                                        optionsParser.getSourcePathList(),
                                        PCHContainerOps),
        optionsParser(optionsParser) {}
  std::vector<ParLoop> &getParLoops() { return loops; }

  /// @brief Generates kernelfiles for all parLoop
  /// Currently only seqkernels created.
  void generateKernelFiles() {
    for (ParLoop &loop : loops) {
      std::string name = loop.getName();
      SeqRefactoringTool tool(optionsParser.getCompilations(), loop);
      if (tool.generateKernelFile()) {
        llvm::outs() << "Error during processing ";
      }
      llvm::outs() << name << "\n";
    }
  }

  /// @brief Create the output files based on the replacements.
  void writeOutReplacements(){
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
  
      std::string filename = getOutputFileName(Sources.getFileEntryForID(I->first));
      std::error_code ec;
      llvm::raw_fd_ostream outfile{llvm::StringRef(filename), ec,
                                   llvm::sys::fs::F_Text | llvm::sys::fs::F_RW};

      I->second.write(outfile);
    }
  }
  
  /// @brief Generate output filename.
  ///
  /// @param Entry The input file that processed.
  ///
  /// @return output filename (xxx_op.cpp)
  virtual std::string getOutputFileName(const clang::FileEntry *Entry ) const {
    llvm::outs() << "Rewrite buffer for file: " << Entry->getName() << "\n";

    std::string filename = Entry->getName().str();
    size_t basename_start = filename.rfind("/") + 1,
           basename_end = filename.rfind(".");
    if (basename_start == std::string::npos)
      basename_start = 0;
    if (basename_end == std::string::npos || basename_end < basename_start)
      llvm::errs() << "Invalid filename: " << Entry->getName() << "\n";
    filename = filename.substr(basename_start, basename_end - basename_start) +
               "_op.cpp";
    return filename;
  }
  
};

} // namespace OP2

#endif /* ifndef OP2REFACTORINGTOOL_HPP */
