#include "utils.h"
#include <clang/Basic/TargetInfo.h>

namespace OP2 {
std::unique_ptr<clang::CompilerInstance> createCompilerInstance() {
  auto Instance = llvm::make_unique<clang::CompilerInstance>();
  Instance->createDiagnostics();
  auto TO = std::make_shared<clang::TargetOptions>();
  TO->Triple = llvm::sys::getProcessTriple();
  Instance->setTarget(
      clang::TargetInfo::CreateTargetInfo(Instance->getDiagnostics(), TO));
  Instance->createFileManager();
  Instance->createSourceManager(Instance->getFileManager());
  Instance->createPreprocessor(clang::TU_Complete);
  Instance->createASTContext();
  return Instance;
}

}
