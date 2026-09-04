# AGENTS.md

This file records the user's long-term preferences for how Codex should write and change code. It is not a project overview; before starting work, still read the task-relevant code, docs, and tests.

## Core Expectations

- Think before coding. Do not assume, do not hide confusion, and surface tradeoffs when there are multiple reasonable interpretations.
- Prefer simplicity. Write the minimum code needed for the current request, with no speculative flexibility, configuration, or future-facing features.
- Make surgical changes. Touch only the files and lines required by the current task; do not opportunistically refactor, reformat, or clean up adjacent code.
- Execute toward verifiable goals. For multi-step work, state a brief plan and define how each step will be checked; actually verify before calling the task done.
- Architecture changes require explicit user approval before implementation.

## Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Ask before finalizing: would a senior engineer say this is overcomplicated? If yes, simplify.

## Surgical Changes

Touch only what is necessary. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting unless it directly supports the request.
- Do not refactor things that are not broken.
- Match the existing style, even if you would choose a different style.
- If unrelated dead code is noticed, mention it rather than deleting it.

When your changes create orphans:
- Remove imports, variables, functions, and tests that your changes made unused.
- Do not remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## Goal-Driven Execution

Transform tasks into verifiable goals:

- "Add validation" means write or identify checks for invalid inputs, then make them pass.
- "Fix the bug" means reproduce or reason from a concrete failure, then verify the fix.
- "Refactor X" means preserve behavior and run relevant checks before calling it done.

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let Codex loop independently. Weak criteria require clarification before implementation.

## Workspace And Git

- Do not create a new branch unless explicitly asked.
- Do not create a new worktree.
- Do not modify files outside the current task target.
- Do not reset, checkout, delete, or overwrite user or externally generated uncommitted changes.
- If the worktree is dirty in files related to the task, understand and adapt to the existing changes; ask only when safe progress is not possible.
