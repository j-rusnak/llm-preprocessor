# Production Configuration

Use `config.production.example.json` for a local loopback production profile.
It binds to `127.0.0.1`, requires local proxy auth, rate-limits callers, caps
request body size, enables graph-backed retrieval, enables structural fast-path
answers, and keeps upstream provider auth separate from local proxy auth.

Use `config.lan.example.json` only as a template when another trusted machine
on the LAN must reach the proxy. It binds to `0.0.0.0`, keeps
`allow_unsafe_remote_proxy: false`, disables forwarding the client
`Authorization` header upstream, and keeps positive rate-limit and request-size
settings. The tracked file intentionally contains a placeholder local bearer
token, so it must be copied to a private local config and given a deployment
token before it can pass health or serve.

## Auth Separation

Local proxy auth and upstream provider auth are separate controls:

- Send local proxy credentials with `X-Preprocessor-Authorization: Bearer <local-token>`.
- Send upstream provider credentials through `upstream_api_key` or a deployment
  secret mechanism.
- Keep `proxy_forward_client_authorization: false` when clients use
  `Authorization` for provider credentials or when local proxy auth is enabled.

Do not commit real provider keys. The tracked examples keep `upstream_api_key`
empty so deployments can inject secrets locally.

## Request Size Limit

Keep `proxy_max_request_bytes` positive for production profiles. The proxy
enforces this limit before JSON parsing or upstream forwarding, which bounds
memory use and rejects oversized requests early.

## Health Check

Run health before serving a profile:

```powershell
.\build\preprocessor_app.exe --health config.production.example.json
Copy-Item config.lan.example.json config.lan.local.json
# Edit config.lan.local.json and replace proxy_auth_bearer_tokens.
.\build\preprocessor_app.exe --health config.lan.local.json
```

Expected prerequisite: the configured embedding model and vocabulary must exist
at `model_path` and `vocab_path`, for example `models/model.onnx` and
`models/vocab.txt`. If those local assets are missing, health may exit non-zero;
install the model and vocab instead of weakening the config.

The tracked LAN template is expected to fail health until its placeholder token
is replaced. Placeholder-looking non-loopback bearer tokens containing markers
such as `replace`, `change-me`, `changeme`, `placeholder`, or `example` are
rejected by `ConfigLoader`.

After health passes, start the proxy:

```powershell
.\build\preprocessor_app.exe --serve config.production.example.json
```

Use the LAN profile only behind network controls appropriate for the deployment.
Never use `allow_unsafe_remote_proxy` for real deployments.
