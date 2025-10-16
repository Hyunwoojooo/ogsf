"""YAML configuration loading with schema validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type

try:  # pragma: no cover - optional dependency guard
    import yaml
except ModuleNotFoundError as exc:
    yaml = None  # type: ignore[assignment]
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency guard
    from pydantic import BaseModel, Field, ValidationError
except ModuleNotFoundError as exc:  # pragma: no cover - hard failure in runtime
    raise RuntimeError("pydantic is required for configuration validation") from exc


class ConfigError(RuntimeError):
    """Base error for configuration loading and validation."""


class ConfigLoadError(ConfigError):
    """Raised when a configuration file cannot be read."""


class ConfigValidationError(ConfigError):
    """Raised when validation against the schema fails."""


class DefaultConfigSchema(BaseModel):
    """Conservative default schema expecting canonical sections."""

    data: Dict[str, Any] = Field(default_factory=dict)
    model: Dict[str, Any] = Field(default_factory=dict)
    train: Dict[str, Any] = Field(default_factory=dict)
    solver: Dict[str, Any] = Field(default_factory=dict)
    runtime: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class ExperimentConfig(Mapping[str, Any]):
    """Dictionary-like wrapper over validated configuration data."""

    def __init__(self, payload: BaseModel) -> None:
        self._model = payload
        self._data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Return value for *key*, falling back to *default* when missing."""
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the configuration as a plain dictionary."""
        return deepcopy(self._data)

    @property
    def model(self) -> BaseModel:
        """Expose the underlying pydantic model for advanced usage."""
        return self._model


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* without mutating inputs."""

    result: Dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = deepcopy(value)
    return result


def _validate_with_schema(schema_cls: Type[BaseModel], payload: Dict[str, Any]) -> BaseModel:
    if hasattr(schema_cls, "model_validate"):
        return schema_cls.model_validate(payload)  # type: ignore[attr-defined]
    return schema_cls.parse_obj(payload)


def load_config(
    path: str | Path,
    *,
    schema: Type[BaseModel] | None = None,
    _stack: Optional[Set[Path]] = None,
) -> ExperimentConfig:
    """Load a YAML configuration file and validate it with *schema*."""
    schema_cls = schema or DefaultConfigSchema
    location = Path(path).resolve()
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configuration files") from YAML_IMPORT_ERROR

    stack = _stack if _stack is not None else set()
    if location in stack:
        raise ConfigError(f"Cyclic configuration inheritance detected for {location}")
    stack.add(location)

    try:
        raw_text = location.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - passthrough read error
        stack.discard(location)
        raise ConfigLoadError(f"Unable to read configuration: {location}") from exc

    payload_raw: Dict[str, Any] = yaml.safe_load(raw_text) or {}

    inherit_spec = payload_raw.pop("inherit", None)
    merged_payload: Dict[str, Any] = {}

    try:
        if inherit_spec:
            inherit_entries = (
                inherit_spec
                if isinstance(inherit_spec, (list, tuple))
                else [inherit_spec]
            )
            for entry in inherit_entries:
                entry_path = Path(entry) if not isinstance(entry, Path) else entry
                if not entry_path.is_absolute():
                    entry_path = (location.parent / entry_path).resolve()
                base_config = load_config(entry_path, schema=schema_cls, _stack=stack)
                merged_payload = _deep_merge(merged_payload, base_config.to_dict())

        merged_payload = _deep_merge(merged_payload, payload_raw)

        validated = _validate_with_schema(schema_cls, merged_payload)
    except ValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc
    finally:
        stack.discard(location)

    return ExperimentConfig(validated)
