"""Standardized sample schema definitions for NLQ."""

from __future__ import annotations

from typing import Any, Mapping, MutableSequence, Sequence, TypedDict

try:  # pragma: no cover - optional dependency
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - numpy optional
    _np = None

__all__ = ["NLQLabel", "NLQSample", "validate"]


class NLQLabel(TypedDict):
    """Temporal label describing a relevant segment."""

    start: float
    end: float


class NLQSample(TypedDict):
    """Schema for a single NLQ training or inference sample."""

    video_id: str
    qid: str
    video_feat: Sequence[Sequence[float]]
    text_feat: Sequence[Sequence[float]]
    labels: MutableSequence[NLQLabel]
    fps: float


_NUMERIC_TYPES = (float, int)
_REQUIRED_KEYS = set(NLQSample.__annotations__.keys())


def _is_float_matrix(value: Any) -> bool:
    if _np is not None and isinstance(value, _np.ndarray):
        return value.ndim == 2 and value.dtype.kind in {"f", "i"}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for row in value:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                return False
            if not all(isinstance(elem, _NUMERIC_TYPES) for elem in row):
                return False
        return True

    return False


def _validate_labels(labels: Any) -> None:
    assert isinstance(labels, Sequence), "labels must be a sequence"
    for idx, item in enumerate(labels):
        assert isinstance(item, Mapping), f"label[{idx}] must be a mapping"
        assert {"start", "end"} <= set(item.keys()), f"label[{idx}] missing keys"
        assert isinstance(item["start"], _NUMERIC_TYPES), f"label[{idx}]['start'] must be numeric"
        assert isinstance(item["end"], _NUMERIC_TYPES), f"label[{idx}]['end'] must be numeric"


def validate(sample: Mapping[str, Any]) -> None:
    """Validate *sample* against the NLQ sample schema."""
    assert isinstance(sample, Mapping), "sample must be a mapping"

    missing = _REQUIRED_KEYS - sample.keys()
    assert not missing, f"missing required keys: {sorted(missing)}"

    assert isinstance(sample["video_id"], str), "video_id must be a string"
    assert isinstance(sample["qid"], str), "qid must be a string"
    assert _is_float_matrix(sample["video_feat"]), "video_feat must be 2D numeric"
    assert _is_float_matrix(sample["text_feat"]), "text_feat must be 2D numeric"
    _validate_labels(sample["labels"])
    assert isinstance(sample["fps"], _NUMERIC_TYPES), "fps must be numeric"
    assert sample["fps"] > 0, "fps must be positive"
