from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai_service_kit.health import (
    BaseHealthCheck,
    CheckResult,
    ComponentKind,
    ComponentStatus,
    HealthStatus,
    NoOpMetricsCollector,
    ProviderDiagnosticsResult,
    ServiceContext,
    VectorStoreDiagnosticsResult,
)

from .config import Settings

SUPPORTED_PROVIDERS = ("openai", "gemini", "claude")
SUPPORTED_VECTORSTORES = ("chroma",)


@dataclass(slots=True)
class ProviderRuntime:
    name: str
    model: str | None
    configured: bool
    supported: bool
    initialized: bool


@dataclass(slots=True)
class VectorStoreRuntime:
    backend: str
    default_collection_name: str
    configured: bool
    supported: bool
    initialized: bool
    collections_count: int


@dataclass(slots=True)
class RagConfiguration:
    chunk_size: int
    chunk_overlap: int
    default_k: int
    default_rerank_k: int
    provider_runtime: ProviderRuntime
    vectorstore_runtime: VectorStoreRuntime


class ConfigurationHealthCheck(BaseHealthCheck):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "configuration"

    async def run(self) -> CheckResult:
        provider_name = self._settings.provider_type.strip().lower()
        vectorstore_backend = self._settings.vectorstore_backend.strip().lower()
        errors: list[str] = []

        if provider_name not in SUPPORTED_PROVIDERS:
            errors.append(f"Unsupported provider: {provider_name}")
        elif not self._settings.selected_provider_api_key():
            errors.append(f"Missing API key for provider: {provider_name}")

        if vectorstore_backend not in SUPPORTED_VECTORSTORES:
            errors.append(f"Unsupported vector store backend: {vectorstore_backend}")

        if self._settings.chunk_size <= 0:
            errors.append(f"Invalid chunk size: {self._settings.chunk_size}")

        if self._settings.chunk_overlap < 0 or self._settings.chunk_overlap >= self._settings.chunk_size:
            errors.append(f"Invalid chunk overlap: {self._settings.chunk_overlap}")

        if not errors:
            status = HealthStatus.HEALTHY
            summary = "Operational configuration is valid"
        elif any(error.startswith("Unsupported") for error in errors):
            status = HealthStatus.CRITICAL
            summary = "Operational configuration contains unsupported components"
        else:
            status = HealthStatus.DEGRADED
            summary = "Operational configuration is incomplete"

        return CheckResult(
            name=self.name,
            status=status,
            summary=summary,
            details={
                "provider_type": provider_name,
                "available_providers": list(SUPPORTED_PROVIDERS),
                "vectorstore_backend": vectorstore_backend,
                "available_vectorstores": list(SUPPORTED_VECTORSTORES),
                "chunk_size": self._settings.chunk_size,
                "chunk_overlap": self._settings.chunk_overlap,
                "default_k": self._settings.default_k,
                "default_rerank_k": self._settings.default_rerank_k,
            },
            errors=tuple(errors),
        )


def _build_provider_runtime(settings: Settings) -> ProviderRuntime:
    provider_name = settings.provider_type.strip().lower()
    supported = provider_name in SUPPORTED_PROVIDERS
    configured = supported and bool(settings.selected_provider_api_key())
    initialized = configured
    return ProviderRuntime(
        name=provider_name,
        model=settings.selected_provider_model(),
        configured=configured,
        supported=supported,
        initialized=initialized,
    )


def _build_vectorstore_runtime(settings: Settings) -> VectorStoreRuntime:
    backend = settings.vectorstore_backend.strip().lower()
    supported = backend in SUPPORTED_VECTORSTORES
    configured = supported and bool(settings.default_collection_name)
    initialized = configured
    return VectorStoreRuntime(
        backend=backend,
        default_collection_name=settings.default_collection_name,
        configured=configured,
        supported=supported,
        initialized=initialized,
        collections_count=1 if configured else 0,
    )


def _provider_status(provider_runtime: ProviderRuntime) -> ComponentStatus:
    if not provider_runtime.supported:
        status = HealthStatus.CRITICAL
        error = f"Unsupported provider: {provider_runtime.name}"
    elif not provider_runtime.configured:
        status = HealthStatus.DEGRADED
        error = f"Missing credentials for provider: {provider_runtime.name}"
    else:
        status = HealthStatus.HEALTHY
        error = None

    return ComponentStatus(
        name=provider_runtime.name,
        kind=ComponentKind.PROVIDER,
        status=status,
        configured=provider_runtime.configured,
        available=provider_runtime.supported,
        initialized=provider_runtime.initialized,
        details={"model": provider_runtime.model},
        error=error,
    )


def _vectorstore_status(vectorstore_runtime: VectorStoreRuntime) -> ComponentStatus:
    if not vectorstore_runtime.supported:
        status = HealthStatus.CRITICAL
        error = f"Unsupported vector store backend: {vectorstore_runtime.backend}"
    elif not vectorstore_runtime.configured:
        status = HealthStatus.DEGRADED
        error = f"Vector store backend is missing required configuration: {vectorstore_runtime.backend}"
    else:
        status = HealthStatus.HEALTHY
        error = None

    return ComponentStatus(
        name=vectorstore_runtime.backend,
        kind=ComponentKind.VECTORSTORE,
        status=status,
        configured=vectorstore_runtime.configured,
        available=vectorstore_runtime.supported,
        initialized=vectorstore_runtime.initialized,
        details={"default_collection_name": vectorstore_runtime.default_collection_name},
        error=error,
    )


def _provider_diagnostics(provider_runtime: ProviderRuntime) -> ProviderDiagnosticsResult:
    provider_status = _provider_status(provider_runtime)
    return ProviderDiagnosticsResult(
        provider=provider_runtime.name,
        status=provider_status.status,
        configured=provider_runtime.configured,
        available=provider_runtime.supported,
        initialized=provider_runtime.initialized,
        models_available=(provider_runtime.model,) if provider_runtime.model else (),
        error=provider_status.error,
        details={"model": provider_runtime.model},
    )


def _vectorstore_diagnostics(vectorstore_runtime: VectorStoreRuntime) -> VectorStoreDiagnosticsResult:
    vectorstore_status = _vectorstore_status(vectorstore_runtime)
    return VectorStoreDiagnosticsResult(
        backend=vectorstore_runtime.backend,
        status=vectorstore_status.status,
        configured=vectorstore_runtime.configured,
        available=vectorstore_runtime.supported,
        initialized=vectorstore_runtime.initialized,
        collections_count=vectorstore_runtime.collections_count,
        default_collection=vectorstore_runtime.default_collection_name,
        error=vectorstore_status.error,
        details={"default_collection_name": vectorstore_runtime.default_collection_name},
    )


def debug_snapshot(context: ServiceContext) -> dict[str, Any]:
    configuration = context.configuration()
    return {
        "service_name": context.service_name,
        "service_version": context.service_version,
        "configuration": {
            "provider": configuration.provider,
            "available_providers": list(configuration.available_providers),
            "vectorstore": configuration.vectorstore,
            "available_vectorstores": list(configuration.available_vectorstores),
            "mock_mode": configuration.mock_mode,
            "debug_mode": configuration.debug_mode,
            "cors_enabled": configuration.cors_enabled,
            "settings": configuration.settings,
        },
        "metrics_collector": type(context.metrics_collector).__name__,
        "health_checks": [getattr(check, "name", type(check).__name__) for check in context.health_checks],
        "diagnostics_checks": [getattr(check, "name", type(check).__name__) for check in context.diagnostics_checks],
    }


def build_service_context(settings: Settings) -> ServiceContext:
    provider_runtime = _build_provider_runtime(settings)
    vectorstore_runtime = _build_vectorstore_runtime(settings)
    metrics_collector = NoOpMetricsCollector()
    configuration_check = ConfigurationHealthCheck(settings)

    rag_configuration = RagConfiguration(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        default_k=settings.default_k,
        default_rerank_k=settings.default_rerank_k,
        provider_runtime=provider_runtime,
        vectorstore_runtime=vectorstore_runtime,
    )

    async def provider_status_resolver() -> tuple[ComponentStatus, ...]:
        return (_provider_status(provider_runtime),)

    async def vectorstore_status_resolver() -> tuple[ComponentStatus, ...]:
        return (_vectorstore_status(vectorstore_runtime),)

    async def provider_diagnostics_resolver() -> tuple[ProviderDiagnosticsResult, ...]:
        return (_provider_diagnostics(provider_runtime),)

    async def vectorstore_diagnostics_resolver() -> tuple[VectorStoreDiagnosticsResult, ...]:
        return (_vectorstore_diagnostics(vectorstore_runtime),)

    async def benchmarks_resolver() -> dict[str, Any]:
        return {
            "bootstrap": {
                "provider_initialized": provider_runtime.initialized,
                "vectorstore_initialized": vectorstore_runtime.initialized,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "default_k": settings.default_k,
                "default_rerank_k": settings.default_rerank_k,
            }
        }

    return ServiceContext(
        service_name=settings.app_name,
        service_version=settings.app_version,
        provider=provider_runtime.name,
        available_providers=SUPPORTED_PROVIDERS,
        vectorstore=vectorstore_runtime.backend,
        available_vectorstores=SUPPORTED_VECTORSTORES,
        mock_mode=settings.mock_mode,
        debug_mode=settings.app_debug,
        cors_enabled=settings.enable_cors,
        masked_secrets=settings.masked_secrets(),
        settings=settings.operational_settings(),
        metrics_collector=metrics_collector,
        health_checks=(configuration_check,),
        diagnostics_checks=(configuration_check,),
        provider_status_resolver=provider_status_resolver,
        vectorstore_status_resolver=vectorstore_status_resolver,
        provider_diagnostics_resolver=provider_diagnostics_resolver,
        vectorstore_diagnostics_resolver=vectorstore_diagnostics_resolver,
        benchmarks_resolver=benchmarks_resolver,
    )