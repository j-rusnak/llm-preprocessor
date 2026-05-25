# Production Proxy Configuration

Run the local proxy on a loopback interface by default. A non-loopback bind such
as `0.0.0.0` exposes the OpenAI-compatible service to the network and must use
local proxy authentication, a positive request-size limit, and explicit review.

Use `X-Preprocessor-Authorization` for local proxy bearer tokens when clients
also need an upstream `Authorization` header for the model provider. Keep
`proxy_forward_client_authorization` disabled in that deployment so the local
bearer token is never forwarded upstream.

The health endpoint may remain public for liveness probes. Protect `/stats`,
sync cache routes, sync vector routes, and chat completions whenever proxy auth
is configured. Do not ship placeholder bearer tokens in production config.
