from slowapi import Limiter
from slowapi.util import get_remote_address


def _get_tenant_or_ip(request) -> str:
    # Rate-limit by tenant_id if it's in request state, otherwise by IP
    tenant_id = getattr(getattr(request, "state", None), "tenant_id", None)
    if tenant_id:
        return str(tenant_id)
    return get_remote_address(request)


limiter = Limiter(key_func=_get_tenant_or_ip, default_limits=["100/minute"])
