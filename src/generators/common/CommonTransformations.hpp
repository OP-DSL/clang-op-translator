#ifndef COMMONTRANSFORMATIONS_HPP
#define COMMONTRANSFORMATIONS_HPP
#include "core/OPParLoopData.h"
#include "core/clang_op_translator_core.h"
#include "core/utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"
#include <functional>
#include <optional>

namespace op_dsl {
namespace matchers = clang::ast_matchers;

/**
 * @brief Baseclass for tansformations on the skeletons.
 *
 * This class provides templated functions holding the boilerplate codes for
 * common transformations on skeletons. All operations over the skeletons
 * subclass this.
 */
class CommonOpration {
protected:
  using repl_map = std::map<std::string, clang::tooling::Replacements>;
  const ParLoop &loop;
  const OPOptimizations &opt;
  repl_map *Replace; /**< Replacement map from refactoring tools. */

public:
  // Initialisation, constructors..
  CommonOpration(const ParLoop &loop, const OPOptimizations &opt,
                 repl_map *Replace) noexcept
      : loop(loop), opt(opt), Replace(Replace) {}
  virtual ~CommonOpration() = default;
  CommonOpration(const CommonOpration &) noexcept = default;
  CommonOpration &operator=(const CommonOpration &) noexcept = default;
  CommonOpration(CommonOpration &&) noexcept = default;
  CommonOpration &operator=(CommonOpration &&) noexcept = default;

  // This operator will be used from the MatchMaker class.
  virtual void
  operator()(const matchers::MatchFinder::MatchResult &Result) const = 0;

protected:
  // Templated functions for different kinds of replacements.

  /// @brief Template function for simple handlers to replace matches with
  /// offset at the end of the match.
  ///
  /// @tparam MatchType Type of the node bind to Key
  /// @tparam Offset Offset to the end of the replacement from the end location
  ///   of the matched ast node
  /// @param Result The Results from the callback
  /// @param Key  Key used in matcher that we want to handle
  /// @param ReplacementGenerator Function that gives back the replacement
  /// string
  ///
  /// @return 0 on success
  template <typename MatchType, int Offset = 0>
  int lineReplHandler(
      const clang::ast_matchers::MatchFinder::MatchResult &Result,
      const std::string &Key,
      std::function<std::optional<std::string>(void)> &&ReplacementGenerator)
      const {
    const auto *match = Result.Nodes.getNodeAs<MatchType>(Key);
    if (!match)
      return 1; // We shouldn't handle this match
    debugs() << Key << "\n";
    clang::SourceManager *sm = Result.SourceManager;
    std::string filename =
        sm->getFilename(sm->getFileLoc(match->getBeginLoc()));
    clang::SourceRange replRange(
        sm->getFileLoc(match->getBeginLoc()),
        sm->getFileLoc(match->getEndLoc()).getLocWithOffset(Offset));
    std::optional<std::string> replacement = ReplacementGenerator();
    if (!replacement.has_value()) {
      return 0; // if the generator returned an empty optional we don't need to
                // create a replacement.
    }
    clang::tooling::Replacement repl(
        *sm, clang::CharSourceRange(replRange, false), *replacement);
    if (llvm::Error err = (*Replace)[filename].add(repl)) {
      // TODO diagnostics..
      debugs() << "Replacement for key: " << Key << " failed in: " << filename
               << "\n";
    }
    return 0;
  }

  /// @brief Template function for simple handlers to replace source ranges
  /// defined by the location of two AST node (the match and endMatch).
  ///
  /// @tparam MatchType Type of the node bind to Key (determine the start of the
  /// replacement)
  /// @tparam EndMatchType Type of the node bind to EndKey (determine the end of
  /// the replacement)
  /// @tparam StartOffset Offset to the start of the replacement from the
  /// location of the match to Key
  /// @tparam EndOffset Offset to the end of the replacement from the location
  ///   of the match to EndKey
  /// @param Result The Results from the callback
  /// @param Key  Key used in matcher that we want to handle
  /// @param ReplacementGenerator Function that gives back the replacement
  /// string
  /// @param EndKey  Key used in to get the AST node with EndMatchType
  ///
  /// @return 0 on success
  template <typename MatchType, typename EndMatchType, int StartOffset = 0,
            int EndOffset = 0>
  int fixEndReplHandler(
      const clang::ast_matchers::MatchFinder::MatchResult &Result,
      const std::string &Key,
      std::function<std::optional<std::string>(void)> &&ReplacementGenerator,
      std::string EndKey = "END") const {
    const auto *match = Result.Nodes.getNodeAs<MatchType>(Key);
    const auto *endMatch = Result.Nodes.getNodeAs<EndMatchType>(EndKey);
    if (!match || !endMatch)
      return 1; // We shouldn't handle this match
    debugs() << Key << "\n";
    clang::SourceManager *sm = Result.SourceManager;
    std::string filename =
        sm->getFilename(sm->getFileLoc(match->getBeginLoc()));
    clang::SourceRange replRange(
        sm->getFileLoc(match->getBeginLoc()).getLocWithOffset(StartOffset),
        sm->getFileLoc(endMatch->getBeginLoc()).getLocWithOffset(EndOffset));
    std::optional<std::string> replacement = ReplacementGenerator();
    if (!replacement.has_value()) {
      return 0; // if the generator returned an empty optional we don't need to
                // create a replacement.
    }
    clang::tooling::Replacement repl(
        *sm, clang::CharSourceRange(replRange, false), *replacement);
    if (llvm::Error err = (*Replace)[filename].add(repl)) {
      // TODO diagnostics..
      llvm::errs() << "Replacement for key: " << Key
                   << " failed in: " << filename << "\n";
    }
    return 0;
  }
};

class KernelFuncDeclaratorOperation : public CommonOpration {
public:
  KernelFuncDeclaratorOperation(const ParLoop &loop, const OPOptimizations &opt,
                                repl_map *Replace) noexcept
      : CommonOpration(loop, opt, Replace) {}

  void
  operator()(const matchers::MatchFinder::MatchResult &Result) const override {
    lineReplHandler<clang::FunctionDecl, 1>(
        Result, key, [this]() -> std::optional<std::string> {
          return {loop.getUserFuncInfo().getInlinedFuncDecl()};
        });
  }
  static inline const std::string key = "user_func";
};

class DeclNumArgOperation : public CommonOpration {
public:
  DeclNumArgOperation(const ParLoop &loop, const OPOptimizations &opt,
                      repl_map *Replace) noexcept
      : CommonOpration(loop, opt, Replace) {}

  void
  operator()(const matchers::MatchFinder::MatchResult &Result) const override {
    lineReplHandler<clang::VarDecl, 2>(
        Result, key, [this]() -> std::optional<std::string> {
          return {"const int num_args = " +
                  std::to_string(loop.getArgs().size()) + ";"};
        });
  }
  static inline const std::string key = "decl_num_arg";
};

class ParLoopDeclOperation : public CommonOpration {
  DSL dsl;

public:
  ParLoopDeclOperation(const ParLoop &loop, const OPOptimizations &opt,
                       repl_map *Replace, DSL dsl) noexcept
      : CommonOpration(loop, opt, Replace), dsl(dsl) {}

  void
  operator()(const matchers::MatchFinder::MatchResult &Result) const override {
    fixEndReplHandler<clang::FunctionDecl, clang::CompoundStmt>(
        Result, key, [this]() -> std::optional<std::string> {
          std::string parLoopDecl = loop.getParLoopDef();
          size_t pos = parLoopDecl.find("DSL");
          if (pos != std::string::npos) {
            parLoopDecl.replace(pos, 3, dsl == OP2 ? "op" : "ops");
          }
          pos = parLoopDecl.find("PARAMS");
          if (pos != std::string::npos) {
            parLoopDecl.replace(pos, 6,
                                dsl == OP2
                                    ? "op_set set"
                                    : "ops_block block, int dim, int *range");
          }
          pos = parLoopDecl.find("ARG");
          while (pos != std::string::npos) {
            parLoopDecl.replace(pos, 3, dsl == OP2 ? "op_arg" : "ops_arg");
            pos = parLoopDecl.find("ARG");
          }
          return {parLoopDecl};
        });
  }
  static inline const std::string key = "par_loop_decl";
};

} // namespace op_dsl

#endif /* ifndef COMMONTRANSFORMATIONS_HPP */
