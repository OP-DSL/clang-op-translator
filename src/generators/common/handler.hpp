#ifndef MATCHHANDLER_FUNC_HPP
#define MATCHHANDLER_FUNC_HPP
#include "core/OPParLoopData.h"
#include "core/utils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"
#include <functional>

namespace OP2 {
using namespace clang::ast_matchers;
using namespace clang;
/// @brief Template function for simple handlers for whole line replacement.
///
/// @tparam MatchType Type of the node bind to Key
/// @tparam Offset Offset to the end of the replacement from the end location
///   of the matched ast node
/// @tparam debug if true print out Key if Results is handled by this function
/// @param Result The Results from the callback
/// @param Replace  Pointer to the map containing all replacements
/// @param Key  Key used in matcher that we want to handle
/// @param ReplacementGenerator Function that gives back the replacement string
///
/// @return 0 on success
template <typename MatchType, int Offset = 0, bool debug = false>
int lineReplHandler(const clang::ast_matchers::MatchFinder::MatchResult &Result,
                    std::map<std::string, tooling::Replacements> *Replace,
                    const std::string Key,
                    std::function<std::string(void)> &&ReplacementGenerator) {
  const MatchType *match = Result.Nodes.getNodeAs<MatchType>(Key);
  if (!match)
    return 1; // We shouldn't handle this match
  if (debug)
    llvm::outs() << Key << "\n";
  clang::SourceManager *sm = Result.SourceManager;
  std::string filename = sm->getFilename(sm->getFileLoc(match->getBeginLoc()));
  SourceRange replRange(
      sm->getFileLoc(match->getBeginLoc()),
      sm->getFileLoc(match->getEndLoc()).getLocWithOffset(Offset));
  std::string replacement = ReplacementGenerator();

  tooling::Replacement repl(*sm, CharSourceRange(replRange, false),
                            replacement);
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement for key: " << Key << " failed in: " << filename
                 << "\n";
  }
  return 0;
}

#define HANDLER(MatchType, Offset, Key, MemberFunction)                        \
  lineReplHandler<MatchType, Offset>(Result, Replace, Key,                     \
                                     std::bind(&MemberFunction, this))

template <typename MatchType, int Offset = 0, bool debug = false>
int fixLengthReplHandler(
    const clang::ast_matchers::MatchFinder::MatchResult &Result,
    std::map<std::string, tooling::Replacements> *Replace,
    const std::string Key, unsigned length,
    std::function<std::string(void)> &&ReplacementGenerator) {
  const MatchType *match = Result.Nodes.getNodeAs<MatchType>(Key);
  if (!match)
    return 1; // We shouldn't handle this match
  if (debug)
    llvm::outs() << Key << "\n";
  clang::SourceManager *sm = Result.SourceManager;
  std::string filename = sm->getFilename(sm->getFileLoc(match->getBeginLoc()));
  std::string replacement = ReplacementGenerator();

  tooling::Replacement repl(
      *sm, sm->getFileLoc(match->getBeginLoc()).getLocWithOffset(Offset),
      length, replacement);
  if (llvm::Error err = (*Replace)[filename].add(repl)) {
    // TODO diagnostics..
    llvm::errs() << "Replacement for key: " << Key << " failed in: " << filename
                 << "\n";
  }
  return 0;
}

template <typename MatchType, typename EndMatchType, int StartOffset = 0,
          int EndOffset = 0, bool debug = false>
int fixEndReplHandler(
    const clang::ast_matchers::MatchFinder::MatchResult &Result,
    std::map<std::string, tooling::Replacements> *Replace,
    const std::string Key,
    std::function<std::string(void)> &&ReplacementGenerator,
    std::string EndKey = "END") {
  const MatchType *match = Result.Nodes.getNodeAs<MatchType>(Key);
  const EndMatchType *endMatch = Result.Nodes.getNodeAs<EndMatchType>(EndKey);
  if (!match || !endMatch)
    return 1; // We shouldn't handle this match
  if (debug)
    llvm::outs() << Key << "\n";
  clang::SourceManager *sm = Result.SourceManager;
  std::string filename = sm->getFilename(sm->getFileLoc(match->getBeginLoc()));
  SourceRange replRange(
      sm->getFileLoc(match->getBeginLoc()).getLocWithOffset(StartOffset),
      sm->getFileLoc(endMatch->getBeginLoc()).getLocWithOffset(EndOffset));
  std::string replacement = ReplacementGenerator();

  if (replacement != "NO_REPL") {
    tooling::Replacement repl(*sm, CharSourceRange(replRange, false),
                              replacement);
    if (llvm::Error err = (*Replace)[filename].add(repl)) {
      // TODO diagnostics..
      llvm::errs() << "Replacement for key: " << Key
                   << " failed in: " << filename << "\n";
    }
  }
  return 0;
}

} // namespace OP2

#endif /* ifndef MATCHHANDLER_FUNC_HPP */
