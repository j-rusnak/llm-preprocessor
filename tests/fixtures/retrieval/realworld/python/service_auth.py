from fastapi import Depends, HTTPException, Request


def verify_bearer_token(request: Request) -> str:
    token = request.headers.get("x-preprocessor-authorization", "")
    if not token.startswith("Bearer "):
        raise HTTPException(status_code=401)
    return token.removeprefix("Bearer ").strip()


def dependency_override_for_tests(
    request: Request, token: str = Depends(verify_bearer_token)
) -> dict[str, str]:
    request.state.local_proxy_token = token
    return {"token": token}
