# Release Checklist

This project is ready to release as local middleware only after the checks below
pass from a clean checkout. A release candidate must not require generated
runtime assets to be committed to Git.

Use [Release Candidate Checklist](RC_CHECKLIST.md) for the exact public branch
and tag sequence.

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
python tools\release_smoke.py
```

Expected result:

- no tracked ignored files,
- `python tools\secret_scan.py` and `python tools\secret_scan.py --ref HEAD`
  report no findings,
- all CTest cases pass,
- smoke runner reports zero failures,
- effectiveness runner exits 0,
- visualizer unit tests pass,
- `--version` reports project version, build config, and the current Git commit,
- install prefix contains `preprocessor_app`, ONNX Runtime redistributables,
  `LLMPreprocessorConfig.cmake`, and `LLMPreprocessorConfigVersion.cmake`,
- package audit passes with no runtime state, model files, or archive artifacts
  in the install prefix,
- `tools\artifact_checksums.py` writes a SHA256 manifest for the install or
  package outputs,
- `tools\release_manifest.py` writes a provenance JSON manifest that records
  the Git commit, version output, platform, artifact paths, sizes, and SHA256
  values.

`tools/release_smoke.py` runs the dashboard browser smoke automatically when
Python Playwright is installed. To run that check directly:

```powershell
python tools\perf_visualizer\tests\validate_dashboard.py `
  --effectiveness-exe build\effectiveness_runner.exe `
  --screenshot-dir benchmarks\results\dashboard-validation
```

Expected result: the dashboard loads, baseline and retrieval diagnostics render,
`Run effectiveness` creates another sample, charts are nonblank, and the mobile
viewport has no horizontal overflow.

If browser validation is not available on a release machine, use
`python tools\release_smoke.py --skip-playwright` and record that exception in
the release notes.

## Config And Secret Checklist

Before using `--serve` outside a local test:

- Copy `config.production.example.json` for loopback-local deployments or
  `config.lan.example.json` for secured non-loopback LAN deployments.
- Keep copied local configs outside Git. Do not commit deployment tokens,
  provider credentials, private model files, archives, or local SQLite state.
- Replace every example local proxy token with a deployment-specific value
  before running health or serve on a copied LAN config.
- Keep `proxy_forward_client_authorization` disabled when local proxy auth uses
  `X-Preprocessor-Authorization`.
- Keep `proxy_max_request_bytes` positive.
- Set `upstream_api_key` through a deployment secret mechanism or local config
  that is not committed.
- Run `preprocessor_app --health <config>` after model and vocab files exist.
- Review [Production Configuration](PRODUCTION_CONFIG.md) before non-loopback
  serving, and do not use `allow_unsafe_remote_proxy` for real deployments.

## Public Push Checklist

Before pushing a release branch to a public repository:

- run `python tools\secret_scan.py`,
- run `python tools\secret_scan.py --ref HEAD` and repeat `--ref` for each
  selected public ref when publishing more than one branch or tag,
- run `python tools\release_smoke.py`,
- confirm `git ls-files -ci --exclude-standard` prints nothing,
- confirm the package audit reports `status: ok`,
- generate and retain `SHA256SUMS` with `python tools\artifact_checksums.py`,
- generate and retain provenance JSON with `python tools\release_manifest.py`,
- confirm GitGuardian or the repository secret-scanning provider is clean after
  the branch is pushed,
- confirm any LAN profile was copied privately and edited outside Git.

## CI Expectations

The release workflow must pass on Windows, Linux, and macOS:

- checkout,
- tracked-ignored-file hygiene gate,
- tracked-file and selected-ref secret/artifact scans,
- release smoke dry run,
- ONNX Runtime download/extraction,
- configure and build,
- install smoke check,
- package audit,
- SHA256 checksum generation,
- release provenance manifest generation,
- CTest,
- smoke runner,
- effectiveness runner,
- visualizer unit tests,
- optional Playwright dashboard smoke when Python Playwright is available.

Do not tag a release if any matrix leg fails.
