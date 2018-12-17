#ifndef LOOPGENERATOR_HPP
#define LOOPGENERATOR_HPP

#include "core/OPParLoopData.h"
#include "core/clang_op_translator_core.h"
#include "generators/common/CommonTransformations.hpp"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"

#include <variant>
#include <vector>

namespace op_dsl {
namespace {
using namespace clang::ast_matchers;
}

class LoopGenerator {
protected:
  using Matcher_t = std::variant<DeclarationMatcher, StatementMatcher>;
  using MMCallback =
      std::pair<Matcher_t, std::unique_ptr<MatchFinder::MatchCallback>>;

  const OPApplication &app;
  const OPOptimizations &opt;
  clang::tooling::CommonOptionsParser &optionsParser;
  const DSL dsl;

  void addBasicOperations(const size_t &loopIdx,
                          clang::tooling::RefactoringTool *tool,
                          std::vector<MMCallback> &matchers) const {
    std::unique_ptr<MatchFinder::MatchCallback> callback(
        new MatchMaker<KernelFuncDeclaratorOperation>(
            KernelFuncDeclaratorOperation(app.getParLoops()[loopIdx], opt,
                                          &tool->getReplacements())));
    matchers.push_back(
        std::move(MMCallback(functionDecl(hasName("kernel"), isDefinition())
                                 .bind(KernelFuncDeclaratorOperation::key),
                             std::move(callback))));
    callback.reset(new MatchMaker<DeclNumArgOperation>(DeclNumArgOperation(
        app.getParLoops()[loopIdx], opt, &tool->getReplacements())));
    matchers.push_back(std::move(
        MMCallback(varDecl(hasName("num_args")).bind(DeclNumArgOperation::key),
                   std::move(callback))));

    callback.reset(new MatchMaker<ParLoopDeclOperation>(ParLoopDeclOperation(
        app.getParLoops()[loopIdx], opt, &tool->getReplacements(), dsl)));
    matchers.push_back(
        std::move(MMCallback(functionDecl(hasName("par_loop_skeleton"),
                                          hasBody(compoundStmt().bind("END")))
                                 .bind(ParLoopDeclOperation::key),
                             std::move(callback))));
  }

  std::string getOutputFileName(const clang::FileEntry *,
                                const size_t loopIdx) const {
    return app.getParLoops()[loopIdx].getName() + "_seqkernel.cpp";
  }

  [[nodiscard]] int generateLoop(size_t loopIdx) const {
    // to chose a skeleton we need a list of skeletons and opts.
    clang::tooling::RefactoringTool tool(
        optionsParser.getCompilations(),
        {"/home/bgd54/itk/work/clang-op-translator/skeletons/"
         "/ops/seq/skeleton_seq.cpp"}); // TODO

    tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
        std::string("-isystem" + std::string(CLANG_SYSTEM_HEADERS) + "/include")
            .c_str()));

    std::vector<MMCallback> matchers; // TODO delete constructors..
    addBasicOperations(loopIdx, &tool, matchers);
    // addSpecificOperations(Finder);

    // Finder
    clang::ast_matchers::MatchFinder Finder;
    // Fill Finder with matchers
    std::for_each(std::begin(matchers), std::end(matchers),
                  [&Finder](const MMCallback &p) {
                    std::visit(
                        [&Finder, &p](const auto &matcher) {
                          Finder.addMatcher(matcher, p.second.get());
                        },
                        p.first);
                  });

    if (int err =
            tool.run(clang::tooling::newFrontendActionFactory(&Finder).get())) {
      return err;
    }
    writeReplacementsTo(
        [this, &loopIdx](const auto &entry) {
          return getOutputFileName(entry, loopIdx);
        },
        &tool);
    return 0;
  }

public:
  LoopGenerator(const OPApplication &app, const OPOptimizations &opt,
                const DSL &dsl,
                clang::tooling::CommonOptionsParser &optionsParser) noexcept
      : app(app), opt(opt), optionsParser(optionsParser), dsl(dsl) {}

  [[nodiscard]] int generate() const {
    for (size_t loopIdx = 0; loopIdx < app.getParLoops().size(); ++loopIdx) {
      if (int err = generateLoop(loopIdx)) {
        return err;
      }
    }
    return 0;
  }
}; // namespace op_dsl

} // namespace op_dsl
#endif /* ifndef LOOPGENERATOR_HPP */
