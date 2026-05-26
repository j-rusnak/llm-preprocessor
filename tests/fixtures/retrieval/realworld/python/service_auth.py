import hmac
import logging

import fastapi


LOGGER = logging.getLogger("preprocessor.service_credentials")
PROXY_CREDENTIAL_HEADER = "x-preprocessor-authorization"
ROUTE_METADATA = ("python", "dependency-override", "bearer-token", "req-state")
CREDENTIAL_CACHE_NAMESPACE = "local-proxy-credentials"


class CredentialSettings:
    def __init__(self, allowed_tokens: frozenset[str], require_loopback: bool = True) -> None:
        self.allowed_tokens = allowed_tokens
        self.require_loopback = require_loopback


def load_credential_settings() -> CredentialSettings:
    return CredentialSettings(allowed_tokens=frozenset({"local-dev-token", "ci-smoke-token"}))


def _extract_bearer_value(raw_header: str) -> str:
    scheme, _, value = raw_header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        raise fastapi.HTTPException(status_code=fastapi.status.HTTP_401_UNAUTHORIZED)
    return value.strip()


def _is_loopback(req) -> bool:
    host = req.client.host if req.client else ""
    return host in {"127.0.0.1", "::1", "localhost"}


def verify_bearer_token(
    req,
    settings: CredentialSettings = fastapi.Depends(load_credential_settings),
) -> str:
    if settings.require_loopback and not _is_loopback(req):
        raise fastapi.HTTPException(status_code=fastapi.status.HTTP_403_FORBIDDEN)
    token = _extract_bearer_value(req.headers.get(PROXY_CREDENTIAL_HEADER, ""))
    if not any(hmac.compare_digest(token, candidate) for candidate in settings.allowed_tokens):
        LOGGER.warning("rejected local credential")
        raise fastapi.HTTPException(status_code=fastapi.status.HTTP_401_UNAUTHORIZED)
    return token


def auth_dependency_override_req_state_for_tests(
    req,
    token: str = fastapi.Depends(verify_bearer_token),
) -> dict[str, str]:
    req.state.local_proxy_token = token
    req.state.request_state_verified = True
    req.state.credential_header = PROXY_CREDENTIAL_HEADER
    return {"token": token, "header": PROXY_CREDENTIAL_HEADER}


def describe_policy(settings: CredentialSettings) -> str:
    return f"tokens:{len(settings.allowed_tokens)} loopback:{settings.require_loopback}"
