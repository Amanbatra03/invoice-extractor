import httpx
from typing import Any


class APIClient:
    def __init__(self, base_url: str, token: str):
        self._base = base_url.rstrip("/")
        self._token = token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    async def _get(self, path: str, params: dict | None = None) -> Any:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base}{path}", headers=self._headers(), params=params)
            if not resp.is_success:
                raise Exception(f"{resp.status_code}: {resp.text}")
            return resp.json()["data"]

    async def _post(self, path: str, json: dict | None = None, files=None) -> Any:
        async with httpx.AsyncClient(timeout=60) as client:
            if files:
                resp = await client.post(
                    f"{self._base}{path}",
                    headers={"Authorization": f"Bearer {self._token}"},
                    files=files,
                )
            else:
                resp = await client.post(f"{self._base}{path}", headers=self._headers(), json=json or {})
            if not resp.is_success:
                raise Exception(f"{resp.status_code}: {resp.text}")
            return resp.json()["data"]

    async def list_invoices(self, page: int = 1, limit: int = 20) -> dict:
        return await self._get("/api/v1/invoices", params={"page": page, "limit": limit})

    async def upload_invoice(self, filename: str, content: bytes, content_type: str) -> dict:
        return await self._post(
            "/api/v1/invoices/upload",
            files={"file": (filename, content, content_type)},
        )

    async def get_invoice(self, invoice_id: str) -> dict:
        return await self._get(f"/api/v1/invoices/{invoice_id}")

    async def delete_invoice(self, invoice_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(
                f"{self._base}/api/v1/invoices/{invoice_id}", headers=self._headers()
            )
            if not resp.is_success:
                raise Exception(f"{resp.status_code}: {resp.text}")
            return resp.json()["data"]

    async def run_extraction(self, invoice_id: str) -> dict:
        return await self._post(f"/api/v1/invoices/{invoice_id}/extract")

    async def get_extraction(self, invoice_id: str) -> dict:
        return await self._get(f"/api/v1/invoices/{invoice_id}/extraction")

    async def ask_question(self, invoice_id: str, question: str) -> dict:
        return await self._post(f"/api/v1/invoices/{invoice_id}/qa", json={"question": question})

    async def compare_invoices(self, invoice_ids: list[str]) -> dict:
        return await self._post("/api/v1/compare", json={"invoice_ids": invoice_ids})

    async def batch_extract(self, invoice_ids: list[str]) -> dict:
        return await self._post("/api/v1/batch/extract", json={"invoice_ids": invoice_ids})

    async def get_job(self, job_id: str) -> dict:
        return await self._get(f"/api/v1/jobs/{job_id}")

    async def list_jobs(self, status: str | None = None, type: str | None = None, limit: int = 50) -> list:
        params: dict = {"limit": limit}
        if status:
            params["status"] = status
        if type:
            params["type"] = type
        return await self._get("/api/v1/jobs", params=params)
