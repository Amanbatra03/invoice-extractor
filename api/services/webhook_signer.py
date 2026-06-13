import hashlib
import hmac
import json
import time


def sign_payload(payload: dict, secret: str) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def verify_signature(payload: dict, secret: str, signature: str) -> bool:
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


def build_webhook_payload(event: str, tenant_id: str, data: dict) -> dict:
    return {
        "event": event,
        "tenant_id": tenant_id,
        "data": data,
        "timestamp": int(time.time()),
    }
