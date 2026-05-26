# Release Checklist

This project is ready to release as local middleware only after the checks below
pass from a clean checkout. A release candidate must not require generated
runtime assets to be committed to Git.

## Supported Release Target

The current release target is a local developer middleware beta/RC:

- OpenAI-compatible proxy bound to loopback by default.
- Optional local bearer or HMAC auth for proxy and sync routes.
- Retrieval, context packing, prompt optimization, streaming, MCP, sync, and
  health-check flows verified through unit, smoke, and effectiveness gates.

This is not an internet-hosted SaaS release target. Non-loopback serving must
use proxy auth, a positive request-size limit, and deployment-specific review.

## Required Local Runtime Assets

Runtime assets stay local or are downloaded in CI:

- ONNX Runtime 1.23.2 extracted under the platform-specific root:
  - `onnxruntime-win-x64-1.23.2`
  - `onnxruntime-linux-x64-1.23.2`
  - `onnxruntime-osx-universal2-1.23.2`
- Embedding model files under `models/`, such as `model.ort` or `model.onnx`
  and `vocab.txt`.
- Local SQLite files such as prompt, vector, and embedding caches.

These paths are intentionally ignored by Git.

## RC Verification

From the repository root on Windows, run in a Visual Studio Developer shell:

```powershell
$trackedIgnored = git ls-files -ci --exclude-standard
if ($trackedIgnored) {
  throw "Tracked ignored files remain:`n$($trackedIgnored -join "`n")"
}

cmake --build build
ctest --test-dir build --output-on-failure
.\build\smoke_runner.exe
.\build\effectiveness_runner.exe
python -m unittest discover tools\perf_visualizer\tests -p "test_*.py"
.\build\preprocessor_app.exe --version
cmake --install build --prefix build\install-check
```

Expected result:

- no tracked ignored files,
- all CTest cases pass,
- smoke runner reports zero failures,
- effectiveness runner exits 0,
- visualizer unit tests pass,
- `--version` reports project version, build config, and commit,
- install prefix contains `preprocessor_app`, ONNX Runtime redistributables,
  and `LLMPreprocessorConfig.cmake`.

If Python Playwright is installed locally, also run:

```powershell
python tools\perf_visualizer\tests\validate_dashboard.py `
  --effectiveness-exe build\effectiveness_runner.exe `
  --screenshot-dir benchmarks\results\dashboard-validation
```

Expected result: the dashboard loads, baseline and retrieval diagnostics render,
`Run effectiveness` creates another sample, charts are nonblank, and the mobile
viewport has no horizontal overflow.

## Config And Secret Checklist

Before using `--serve` outside a local test:

- Copy `config.example.json` and replace `replace-with-local-proxy-token`.
- Keep `proxy_forward_client_authorization` disabled when local proxy auth uses
  `X-Preprocessor-Authorization`.
- Keep `proxy_max_request_bytes` positive.
- Set `upstream_api_key` through a deployment secret mechanism or local config
  that is not committed.
- Run `preprocessor_app --health <config>` after model and vocab files exist.

## CI Expectations

The release workflow must pass on Windows, Linux, and macOS:

- checkout,
- tracked-ignored-file hygiene gate,
- ONNX Runtime download/extraction,
- configure and build,
- install smoke check,
- CTest,
- smoke runner,
- effectiveness runner,
- visualizer unit tests,
- optional Playwright dashboard smoke when Python Playwright is available.

Do not tag a release if any matrix leg fails.
