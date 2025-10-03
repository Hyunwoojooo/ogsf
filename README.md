# EM Module Overview

The **em** package hosts the experiment-management stack for GroundNLQ. It
collects model definitions, loss functions, dataset I/O, metric utilities, and
training scripts needed to reproduce MVP/OGSF experiments while remaining light
enough to run in constrained environments.

## Layout
- `common/` – Shared configuration helpers (Pydantic loader, registry hooks).
- `io/` – Sample schema, WebDataset shard tooling, and streaming reader.
- `features/` – Video/text/object feature loaders plus serialization helpers.
- `models/` – Core architectures (`mvp`, `ogsf`), layers, heads, and losses.
- `metrics/` – Recall@IoU calculators, Soft-NMS, and TTA aggregation.
- `objects/` – YOLO detection wrappers, ByteTrack/DeepSORT tracking, ROI
  feature extraction.
- `train/` – Optimizer/scheduler factories, AMP helpers, accelerate entry, and
  dependency-injected training loops.
- `scripts/` – Thin CLIs for shard building, object extraction/tracking, and
  submission packaging.
- `tests/` – Unit tests and smoke checks for the modules listed above.

## Quick Start
1. (Optional) Duplicate `configs/paths.example.yaml` to `configs/paths.yaml`
   and adjust NAS paths for raw data, features, and shard locations.
2. Choose an experiment preset:
   - `configs/nlq_mvp.yaml` for the cross-attention MVP baseline.
   - `configs/nlq_ogsf.yaml` for the gated multi-scale OGSF variant.
3. For distributed training, edit `configs/accelerate.yaml` or supply your own
   accelerate configuration when launching.
4. Run feature conversions or track building via the helpers in `em/scripts/`
   before starting long experiments.

## Testing
- Run `pytest em/tests -k '<pattern>'` from a full Python environment with
  sufficient shared memory. In the Codex CLI sandbox, shared-memory limits can
  trigger `SIGABRT`; execute tests on a local terminal or container instead.
- Key suites:
  - `em/tests/test_losses.py`, `em/tests/test_metrics.py` – numerical checks.
  - `em/tests/test_mvp.py`, `em/tests/test_ogsf.py` – model forward/backward
    smoke tests.
  - `em/tests/test_scripts.py`, `em/tests/test_train_smoke.py` – CLI and loop
    coverage.

## Notes
- All modules avoid heavy external dependencies. Optional integrations (e.g.
  bitsandbytes, accelerate, torch CUDA AMP) degrade gracefully when unavailable.
- Large dataset artifacts should reside outside the repository; point configs to
  NAS or local mirrors via environment variables (e.g. `WDS_SHARDS`).

