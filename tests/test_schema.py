"""Tests covering NLQ schema definitions and configuration loading."""

from __future__ import annotations

import copy

import pytest

yaml = pytest.importorskip('yaml')

from em.common import config
from em.io import schema

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - handled by skip marker
    BaseModel = None  # type: ignore[assignment]


def _make_valid_sample() -> schema.NLQSample:
    return schema.NLQSample(
        video_id="vid-001",
        qid="q-123",
        video_feat=[[0.1, 0.2], [0.3, 0.4]],
        text_feat=[[0.5, 0.6]],
        labels=[{"start": 1.0, "end": 2.5}],
        fps=30.0,
    )


def test_schema_annotations():
    required = {"video_id", "qid", "video_feat", "text_feat", "labels", "fps"}
    assert required <= set(schema.NLQSample.__annotations__.keys())


def test_validate_accepts_valid_sample():
    sample = _make_valid_sample()
    schema.validate(sample)


def test_validate_missing_key_raises():
    sample = _make_valid_sample()
    sample_missing = copy.deepcopy(sample)
    sample_missing.pop("qid")
    with pytest.raises(AssertionError, match="missing required keys"):
        schema.validate(sample_missing)


def test_validate_type_mismatch_raises():
    sample = _make_valid_sample()
    sample_invalid = copy.deepcopy(sample)
    sample_invalid["fps"] = "30"
    with pytest.raises(AssertionError, match="fps must be numeric"):
        schema.validate(sample_invalid)


if BaseModel is not None:

    class SolverSettings(BaseModel):
        batch_size: int = Field(default=16, ge=1)

        class Config:
            extra = "forbid"

    class ModelSettings(BaseModel):
        name: str
        hidden_dim: int

        class Config:
            extra = "forbid"

    ModelSettings.update_forward_refs()

    class PipelineConfig(BaseModel):
        model: ModelSettings
        solver: SolverSettings = Field(default_factory=SolverSettings)

        class Config:
            extra = "forbid"
else:  # pragma: no cover - ensures names exist for type checkers
    SolverSettings = object  # type: ignore[assignment]
    ModelSettings = object  # type: ignore[assignment]
    PipelineConfig = object  # type: ignore[assignment]


@pytest.mark.skipif(BaseModel is None, reason="pydantic is required for config tests")
class TestConfigLoading:
    """Collection of tests covering em.common.config behaviour."""

    def test_load_config_populates_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model:\n  name: nlq-mvp\n  hidden_dim: 256\n",
            encoding="utf-8",
        )

        loaded = config.load_config(config_file, schema=PipelineConfig)

        assert loaded["model"]["name"] == "nlq-mvp"
        assert loaded["solver"]["batch_size"] == 16

    def test_invalid_type_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model:\n  name: nlq-mvp\n  hidden_dim: not-an-int\n",
            encoding="utf-8",
        )

        with pytest.raises(config.ConfigValidationError):
            config.load_config(config_file, schema=PipelineConfig)

    def test_unknown_key_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model:\n  name: nlq-mvp\n  hidden_dim: 256\nextra: true\n",
            encoding="utf-8",
        )

        with pytest.raises(config.ConfigValidationError):
            config.load_config(config_file, schema=PipelineConfig)
