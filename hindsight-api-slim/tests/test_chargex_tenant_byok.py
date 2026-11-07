"""Focused isolation and BYOK checks for the ChargeX Hindsight extensions."""
from __future__ import annotations

import pytest

from hindsight_api.extensions.chargex import (
    ChargeXLLMRequestExtension,
    ChargeXTenantExtension,
    _valid_schema,
)
from hindsight_api.extensions.llm_request import LLMRequestRefused, LLMRequestTarget
from hindsight_api.extensions.tenant import AuthenticationError
from hindsight_api.models import RequestContext

_SETTINGS = {
    "url": "http://tenant-management.railway.internal:8000",
    "service_key": "test-only-memory-service-key-32-bytes",
}


class StubService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        return self.result


class StubContext:
    def __init__(self):
        self.schemas = []

    async def run_migration(self, schema):
        self.schemas.append(schema)

    def get_memory_engine(self):
        return type("TestMemoryEngine", (), {"_initialized": False})()


def test_reject_untrusted_memory_schemas():
    assert _valid_schema("mem_t000001") == "mem_t000001"
    for wrong in ("public", "mem_t000001;drop", "mem_t000002.extra", "mem_t9999", ""):
        with pytest.raises(ValueError):
            _valid_schema(wrong)


@pytest.mark.asyncio
async def test_authenticated_tenant_maps_only_to_own_schema():
    ext = ChargeXTenantExtension(_SETTINGS)
    svc = StubService({"tenant_id": "T000001", "schema": "mem_t000001"})
    ext._service = svc
    context = StubContext()
    ext.set_context(context)
    caller = RequestContext(api_key="server-signed-token")
    resolved = await ext.authenticate(caller)
    assert resolved.schema_name == "mem_t000001"
    assert caller.tenant_id == "T000001"
    await ext.authenticate(RequestContext(api_key="new-token"))
    assert context.schemas == ["mem_t000001"]
    assert svc.calls[0][1] == "/api/internal/hindsight/validate"


@pytest.mark.asyncio
async def test_untrusted_cross_tenant_mapping_is_denied():
    ext = ChargeXTenantExtension(_SETTINGS)
    ext._service = StubService({"tenant_id": "T000001", "schema": "mem_t000002"})
    ext.set_context(StubContext())
    with pytest.raises(AuthenticationError):
        await ext.authenticate(RequestContext(api_key="server-signed-token"))


@pytest.mark.asyncio
async def test_worker_discovery_does_not_migrate_before_pool_setup():
    ext = ChargeXTenantExtension(_SETTINGS)
    ext._service = StubService({"tenants": [{"tenant_id": "T000001", "schema": "mem_t000001"}]})
    context = StubContext()
    ext.set_context(context)
    tenants = await ext.list_tenants()
    assert [(t.schema, t.tenant_id) for t in tenants] == [("mem_t000001", "T000001")]
    assert context.schemas == []


@pytest.mark.asyncio
async def test_llm_uses_only_selected_tenant_odoo_provider_key():
    ext = ChargeXLLMRequestExtension(_SETTINGS)
    svc = StubService({"provider": "gemini", "model": "gemini-2.5-flash", "api_key": "tenant-secret"})
    ext._service = svc
    target = LLMRequestTarget(
        schema="mem_t000002", bank_id="policies", operation="consolidation",
        provider="openai-responses", model="gpt-6.1-terra"
    )
    result = await ext.resolve(target)
    assert result.llm.provider == "gemini"
    assert result.llm.model == "gemini-2.5-flash"
    assert result.llm.api_key == "tenant-secret"
    assert svc.calls == [("GET", "/api/internal/hindsight/llm-config/T000002", None)]


@pytest.mark.asyncio
async def test_llm_fails_closed_for_public_missing_key_and_unknown_provider():
    ext = ChargeXLLMRequestExtension(_SETTINGS)
    ext._service = StubService({"provider": "openai", "model": "gpt-6.1-terra", "api_key": ""})
    for schema, bank in (("public", "chargex-test"), ("mem_t000001", None), ("mem_t000001", "policies")):
        with pytest.raises(LLMRequestRefused):
            await ext.resolve(LLMRequestTarget(
                schema=schema, bank_id=bank, operation="retain", provider="openai", model="test"
            ))
    ext._service = StubService({"provider": "odoo", "model": "unknown", "api_key": "secret"})
    with pytest.raises(LLMRequestRefused):
        await ext.resolve(LLMRequestTarget(
            schema="mem_t000001", bank_id="policies", operation="reflect", provider="openai", model="test"
        ))
