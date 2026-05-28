# Release Candidate Checklist

Use this checklist for public RC branches and tags. Do not publish a branch or
tag until every local and CI gate below is green from a clean checkout.

## Public Branch And Tag Sequence

1. Start from a clean private working tree:

   ```powershell
   git status --short
   git fetch --all --tags --prune
   git switch polish-features
   git pull --ff-only
   ```

2. Create the release candidate branch:

   ```powershell
   git switch -c release/rc-<version>
   ```

3. Run local history and package hygiene gates:

   ```powershell
   python -m unittest discover tools\tests -p "test_*.py"
   python tools\secret_scan.py
   python tools\secret_scan.py --ref HEAD
   python tools\release_smoke.py --dry-run --skip-playwright --skip-install
   python tools\release_smoke.py
   ```

4. Push the RC branch only after local gates pass:

   ```powershell
   git push origin release/rc-<version>
   ```

5. Wait for the full GitHub Actions matrix to pass on Windows, Linux, and macOS.
   Do not tag from a failed, skipped, or partially rerun matrix.

6. Create and push the signed RC tag from the exact passing branch commit:

   ```powershell
   git rev-parse HEAD
   git tag -s v<version>-rc.<n> -m "llm-preprocessor v<version>-rc.<n>"
   git push origin v<version>-rc.<n>
   ```

7. Verify the public branch and tag point at the same commit:

   ```powershell
   git ls-remote origin refs/heads/release/rc-<version> refs/tags/v<version>-rc.<n>
   ```

8. Build release artifacts from the tag, then generate provenance checksums:

   ```powershell
   git switch --detach v<version>-rc.<n>
   python tools\release_smoke.py
   python tools\artifact_checksums.py build\install-check --output build\install-check.SHA256SUMS
   ```

9. Publish the RC only with:
   - the source tag,
   - install or package outputs produced from that tag,
   - the matching `SHA256SUMS` manifest,
   - release notes that list any skipped optional Playwright validation.

## Required Hygiene Gates

- `python tools\secret_scan.py` must report no tracked secrets, tracked ignored
  files, model files, ONNX archives, SQLite state, extracted ONNX Runtime
  folders, or other generated release artifacts.
- `python tools\secret_scan.py --ref HEAD` must pass for the commit being
  pushed. For a multi-branch public push, repeat `--ref` for each public ref.
- `python tools\package_audit.py <install-prefix>` must report `status: ok`.
- `python tools\artifact_checksums.py <install-prefix> --output <manifest>`
  must generate the checksum file shipped with the RC artifacts.

## CI Release Gate

The public RC branch must pass:

- tracked ignored file hygiene,
- tracked file and selected-ref secret/artifact scan,
- release smoke dry run,
- ONNX Runtime extraction from a clean download directory,
- configure, build, and install,
- package audit,
- SHA256 checksum generation,
- CTest,
- smoke runner,
- effectiveness runner,
- visualizer unit tests,
- optional Playwright dashboard smoke when Python Playwright is available.
