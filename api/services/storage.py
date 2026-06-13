import hashlib
from supabase import create_client, Client
from api.config import get_settings


def _get_client() -> Client:
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def _bucket(tenant_id: str) -> str:
    return f"invoices-{tenant_id}"


def sha256_file(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def upload_file(tenant_id: str, file_name: str, content: bytes) -> str:
    client = _get_client()
    bucket = _bucket(tenant_id)
    try:
        client.storage.create_bucket(bucket, options={"public": False})
    except Exception:
        pass
    storage_path = f"{tenant_id}/{file_name}"
    client.storage.from_(bucket).upload(
        storage_path, content, {"content-type": "application/octet-stream", "upsert": "true"}
    )
    return storage_path


def get_signed_url(tenant_id: str, storage_path: str, expires_in: int = 900) -> str:
    client = _get_client()
    bucket = _bucket(tenant_id)
    result = client.storage.from_(bucket).create_signed_url(storage_path, expires_in)
    return result["signedURL"]


def download_file(tenant_id: str, storage_path: str) -> bytes:
    client = _get_client()
    bucket = _bucket(tenant_id)
    return client.storage.from_(bucket).download(storage_path)


def delete_file(tenant_id: str, storage_path: str) -> None:
    client = _get_client()
    bucket = _bucket(tenant_id)
    client.storage.from_(bucket).remove([storage_path])
