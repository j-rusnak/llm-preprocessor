# AGENTS.md

Guidance for coding agents working in this repository.

## Project Goal

`llm-preprocessor` is a C++17 local middleware for AI coding agents. Its main job
is to reduce token usage and latency by indexing a repo, retrieving focused code
context, optimizing prompts, and forwarding OpenAI-compatible requests to an
upstream model. Treat security, protocol compatibility, and low latency as core
product requirements.

## Work Rules

- Prefer small production slices with tests first.
- Do not make broad refactors while implementing a feature slice.
- Do not revert user changes or unrelated worktree changes.
- Use `rg` for search.
- Use `apply_patch` for manual file edits.
- Keep defaults backward compatible unless the task explicitly changes public
  behavior.
- Never expose a non-loopback proxy without auth unless an explicit unsafe flag
  is part of the requested behavior.

## Build And Verify

Primary verification commands:

```powershell
cmake --build build
.\build\preprocessor_tests.exe --gtest_filter=<FocusedSuiteOrTest>
ctest --test-dir build --output-on-failure
.\build\smoke_runner.exe
.\build\effectiveness_runner.exe
```

Run focused tests during development, then full `ctest`, smoke, and
effectiveness before claiming completion.

## Key Runtime Paths

- `src/openai_proxy.cpp` is the main OpenAI-compatible proxy path.
- `src/config_loader.cpp` owns JSON config parsing and safety validation.
- `src/main.cpp` maps config into runtime objects.
- `tests/test_openai_proxy.cpp` is the main proxy regression suite.
- `tests/test_config_loader.cpp` covers config compatibility and validation.

## Current Production Priorities

1. Clean tracked generated artifacts and package distribution.
2. Improve retrieval quality with stronger symbol extraction and graph-aware
   lookup quality.
3. Add protocol edge cases: upstream error propagation and client disconnect
   handling.
4. Package distribution defaults and production config examples.

## Security Notes

- Local proxy auth and upstream provider auth are separate concerns.
- Prefer `X-Preprocessor-Authorization` for local proxy tokens when clients also
  need `Authorization` for upstream providers.
- `/healthz` may stay public; `/stats` and sync/team routes should be protected
  when proxy auth is configured.
- Enforce request-size limits before JSON parsing or upstream forwarding.
