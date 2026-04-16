from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ModelArtifact:
    symbol: str
    market: str | None
    interval: str
    model_name: str
    training_type: str  # lora | linear_probe | full_ft
    version: str
    status: str
    artifact_path: str
    metrics: dict[str, Any] | None = None
    params: dict[str, Any] | None = None


class ModelRegistryPort(Protocol):
    def upsert(self, item: ModelArtifact) -> None: ...
    def find_ready(
            self,
            symbol: str,
            interval: str,
            model_name: str,
            training_type: str,
            market: str | None = None,
    ) -> list[ModelArtifact]: ...


class ProviderStatePort(Protocol):
    def set_state(self, provider: str, key: str, value: dict[str, Any]) -> None: ...
    def get_state(self, provider: str, key: str) -> dict[str, Any] | None: ...


class UnitOfWorkPort(Protocol):
    model_registry: ModelRegistryPort
    provider_state: ProviderStatePort

    def __enter__(self) -> "UnitOfWorkPort": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...