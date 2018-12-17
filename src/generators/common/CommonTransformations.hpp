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
  /// @tparam debug if true print out Key if Results is handled by this function
  /// @param Result The Results from the callback
  /// @param Key  Key used in matcher that we want to handle
  /// @param ReplacementGenerator Function that gives back the replacement
  /// string
  ///
  /// @return 0 on success
  template <typename MatchType, int Offset = 0, bool debug = false>
  int lineReplHandler(
      const clang::ast_matchers::MatchFinder::MatchResult &Result,
      const std::string &Key,
      std::function<std::optional<std::string>(void)> &&ReplacementGenerator)
      const {
    const auto *match = Result.Nodes.getNodeAs<MatchType>(Key);
    if (!match)
      return 1; // We shouldn't handle this match
    if constexpr (debug)
      llvm::outs() << Key << "\n";
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

} // namespace op_dsl

#endif /* ifndef COMMONTRANSFORMATIONS_HPP */
