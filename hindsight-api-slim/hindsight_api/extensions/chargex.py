"""ChargeX Accounting BYOK integration for Hindsight.

A verified ChargeX tenant maps to one PostgreSQL schema.  The built-in
LLMRequestExtension resolves every LLM call, including background jobs,
through the tenant's existing Odoo AI settings. No customer provider key is
stored in Hindsight or in Railway environment variables.

These extensions deliberately fail closed: never fall back to the platform's
API key if Tenant Management or the customer's provider is unavailable.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
from urllib.parse import urlsplit

import aiohttp

from hindsight_api.config import LLMMemberConfig
from hindsight_api.extensions.llm_request import (
    LLMRequestExtension,
    LLMRequestOverrides,
    LLMRequestRefused,
    LLMRequestTarget,
)
from hindsight_api.extensions.tenant import (
    AuthenticationError,
    Tenant,
    TenantContext,
    TenantExtension,
)
from hindsight_api.models import RequestContext

_SCHEMA = re.compile(r"mem_t[0-9]{6,12}\Z")
_TENANT = re.compile(r"T[0-9]{6,12}\Z")
_PROVIDERS = {"openai": "openai-responses", "gemini": "gemini", "anthropic": "anthropic"}


def _valid_schema(value: object) -> str:
    if not isinstance(value, str) or not _SCHEMA.fullmatch(value):
        raise ValueError("Invalid tenant memory schema")
    return value


def _valid_tenant(value: object) -> str:
    if not isinstance(value, str) or not _TENANT.fullmatch(value):
        raise ValueError("Invalid tenant identity")
    return value


class _ChargeXService:
    """Private, authenticated link to ChargeX Tenant Management."""

    def __init__(self, config: dict[str, str]):
        url = (config.get("url") or "").rstrip("/")
        key = config.get("service_key") or ""
        parsed = urlsplit(url)
        local = parsed.hostname in {"localhost", "127.0.0.1"} and config.get("allow_local") == "true"
        private = parsed.hostname is not None and parsed.hostname.endswith(".railway.internal")
        if (parsed.scheme != "http" or not (private or local) or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                not key or len(key) < 32):
            raise ValueError("ChargeX internal service URL or authentication is not configured securely")
        self.url = url
        self.key = key

    async def request(self, method: str, path: str, body: dict | None = None) -> dict:
        timeout = aiohttp.ClientTimeout(total=12, connect=4)
        headers = {"X-ChargeX-Hindsight-Key": self.key}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, self.url + path, json=body, headers=headers) as response:
                    if response.status != 200:
                        raise RuntimeError(f"ChargeX memory service refused request (HTTP {response.status})")
                    data = await response.json(content_type="application/json")
                    if not isinstance(data, dict):
                        raise RuntimeError("Invalid ChargeX memory service reply")
                    return data
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RuntimeError("ChargeX memory service unavailable") from exc


class ChargeXTenantExtension(TenantExtension):
    """Authenticate through the control plane; isolate tenants by PostgreSQL schema."""

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self._service = _ChargeXService(config)
        # The existing Control Plane may manage ONLY the legacy public/test
        # schema. A platform admin key never authorizes a customer schema.
        self._admin_api_key = config.get("admin_api_key") or os.getenv("HINDSIGHT_CP_DATAPLANE_API_KEY") or ""
        if self._admin_api_key and len(self._admin_api_key) < 24:
            raise ValueError("Invalid Hindsight admin API key")
        self._created: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    async def _ensure_schema(self, schema: str) -> None:
        schema = _valid_schema(schema)
        if schema in self._created:
            return
        lock = self._locks.setdefault(schema, asyncio.Lock())
        async with lock:
            if schema not in self._created:
                # The built-in migration helper uses a PostgreSQL advisory lock
                # to coordinate multiple API/worker replicas.
                await self.context.run_migration(schema)
                self._created.add(schema)

    async def authenticate(self, context: RequestContext) -> TenantContext:
        if not context.api_key:
            raise AuthenticationError("Missing ChargeX tenant credential")
        if self._admin_api_key and secrets.compare_digest(context.api_key, self._admin_api_key):
            return TenantContext(schema_name="public")
        try:
            data = await self._service.request(
                "POST", "/api/internal/hindsight/validate", {"token": context.api_key}
            )
            tenant_id = _valid_tenant(data.get("tenant_id"))
            schema = _valid_schema(data.get("schema"))
            if schema != "mem_" + tenant_id.lower():
                raise ValueError("Tenant/schema mismatch")
            await self._ensure_schema(schema)
        except (ValueError, RuntimeError) as exc:
            raise AuthenticationError("Invalid or unavailable ChargeX tenant") from exc
        context.tenant_id = tenant_id
        return TenantContext(schema_name=schema)

    async def list_tenants(self) -> list[Tenant]:
        data = await self._service.request("GET", "/api/internal/hindsight/tenants")
        raw = data.get("tenants")
        if not isinstance(raw, list):
            raise RuntimeError("Invalid ChargeX tenant directory")
        tenants: list[Tenant] = []
        for item in raw:
            if not isinstance(item, dict):
                raise RuntimeError("Invalid ChargeX tenant directory")
            tenant_id = _valid_tenant(item.get("tenant_id"))
            schema = _valid_schema(item.get("schema"))
            if schema != "mem_" + tenant_id.lower():
                raise RuntimeError("ChargeX tenant/schema mismatch")
            # Do not invoke migrations here: startup calls list_tenants()
            # before the engine connection pool exists and migrates the
            # returned schemas itself. The API lazily migrates newcomers when
            # their first authenticated request arrives.
            tenants.append(Tenant(schema=schema, tenant_id=tenant_id))
        # Startup's schema migration phase runs before the connection pool is
        # initialized, so it consumes the list without extension-side DDL.
        # Once running, the worker's periodic discovery provisions new tenants
        # idempotently, including tenants that have not sent a first request.
        memory = self.context.get_memory_engine()
        if getattr(memory, "_initialized", False):
            for tenant in tenants:
                await self._ensure_schema(tenant.schema)
        if self._admin_api_key:
            tenants.append(Tenant(schema="public"))
        return tenants


class ChargeXLLMRequestExtension(LLMRequestExtension):
    """Use the tenant's Odoo AI provider and key for *every* LLM operation."""

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self._service = _ChargeXService(config)

    async def resolve(self, target: LLMRequestTarget) -> LLMRequestOverrides:
        # No platform-key fallback: public and unknown schemas must never bill
        # the shared deployment's OpenAI account, even in scheduled maintenance.
        try:
            schema = _valid_schema(target.schema)
            tenant_id = _valid_tenant(schema[4:].upper())
            if target.bank_id is None:
                raise ValueError("Unattributed bank")
            data = await self._service.request(
                "GET", "/api/internal/hindsight/llm-config/" + tenant_id
            )
            provider = _PROVIDERS.get(data.get("provider"))
            model = data.get("model")
            key = data.get("api_key")
            if (provider is None or not isinstance(model, str) or not model.strip()
                    or len(model) > 128 or not isinstance(key, str) or not key.strip()
                    or len(key) > 4096):
                raise ValueError("Tenant AI provider not configured")
            return LLMRequestOverrides(
                llm=LLMMemberConfig(provider=provider, model=model.strip(), api_key=key.strip())
            )
        except (RuntimeError, ValueError) as exc:
            raise LLMRequestRefused(
                "ChargeX tenant model credentials unavailable; no shared-key fallback"
            ) from exc
