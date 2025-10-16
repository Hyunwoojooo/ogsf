"""Inference CLI for NLQ models using precomputed WebDataset shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from ..common import config as config_utils
from ..metrics import postprocess, recall_iou
from ..train.accelerate_entry import (
    _build_model,
    _dataset_iterator,
    _expand_path,
    _load_paths,
)

DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)
DEFAULT_RECALL_K = (1, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLQ inference and metric evaluation.")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    parser.add_argument("--paths", help="Optional dataset paths YAML")
    parser.add_argument("--checkpoint", help="Checkpoint to load; defaults to OUTPUT_DIR/<config_name>/model_last.pth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, help="Batch size for inference (defaults to solver.batch_size)")
    parser.add_argument("--topk", type=int, default=5, help="Number of proposals to keep per sample (<=0 keeps all)")
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5, help="Soft-NMS sigma (negative disables Soft-NMS)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for Recall@K metrics")
    return parser.parse_args()


def _resolve_checkpoint(args: argparse.Namespace, config_path: Path) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)

    output_root = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
    default_path = output_root / config_path.stem / "model_last.pth"
    if not default_path.exists():
        raise FileNotFoundError(f"Checkpoint not provided and default path missing: {default_path}")
    return default_path


def _select_pattern(data_cfg: Dict[str, Any], split: str) -> str:
    key = f"{split}_glob"
    pattern = data_cfg.get(key)
    if not pattern:
        if split == "val" and data_cfg.get("shard_glob"):
            pattern = data_cfg["shard_glob"]
        elif split == "train" and data_cfg.get("shard_glob"):
            pattern = data_cfg["shard_glob"]
    if not pattern:
        raise ValueError(f"No shard glob configured for split '{split}'")
    expanded = _expand_path(pattern)
    if not expanded:
        raise ValueError(f"Unable to resolve shard glob for split '{split}'")
    return expanded


def _prepare_outputs(config_path: Path, split: str, *, output_dir: Path | None) -> Dict[str, Path]:
    if output_dir is None:
        output_root = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
        run_dir = output_root / config_path.stem
    else:
        run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    submissions_dir = Path("submissions")
    submissions_dir.mkdir(parents=True, exist_ok=True)

    return {
        "predictions": run_dir / f"{split}_predictions.json",
        "metrics": run_dir / f"{split}_metrics.json",
        "submission": submissions_dir / f"{config_path.stem}_{split}.json",
    }


def _format_segments(predictions: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    formatted: List[Dict[str, float]] = []
    for item in predictions:
        formatted.append(
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "score": float(item.get("score", 1.0)),
            }
        )
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NLQ inference and metric evaluation.")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    parser.add_argument("--paths", help="Optional dataset paths YAML")
    parser.add_argument("--checkpoint", help="Checkpoint to load; defaults to OUTPUT_DIR/<config_name>/model_last.pth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, help="Batch size for inference (defaults to solver.batch_size)")
    parser.add_argument("--topk", type=int, default=5, help="Number of proposals to keep per sample (<=0 keeps all)")
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5, help="Soft-NMS sigma (negative disables Soft-NMS)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for Recall@K metrics")
    parser.add_argument("--output-dir", type=Path, help="Optional directory to store predictions/metrics")
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device (e.g. cuda, cuda:0, cpu). 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Abort when CUDA is unavailable instead of silently falling back to CPU.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    _load_paths(args.paths)
    exp_config = config_utils.load_config(config_path).to_dict()
    model_cfg = exp_config.get("model", {})
    solver_cfg = exp_config.get("solver") or exp_config.get("train", {})
    data_cfg: Dict[str, Any] = exp_config.get("data", {})

    checkpoint_path = _resolve_checkpoint(args, config_path)
    pattern = _select_pattern(data_cfg, args.split)

    batch_size = args.batch_size or int(solver_cfg.get("batch_size", 4))
    sequence_length = int(solver_cfg.get("sequence_length", data_cfg.get("sequence_length", 16)))
    text_tokens = int(data_cfg.get("text_tokens", 8))

    def _resolve_device(request: str) -> torch.device:
        if request == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        device_candidate = torch.device(request)
        if device_candidate.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested via --device={request} but no CUDA devices were detected.")
        return device_candidate

    device = _resolve_device(args.device)
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("CUDA is required (--require-gpu) but no GPU devices were detected.")
    print(f"[infer_nlq] using device: {device}")

    model = _build_model(model_cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "model_state" in state:
            state = state["model_state"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if isinstance(state, dict):
        if all(key.startswith("_orig_mod.") for key in state.keys()):
            state = {key.split("_orig_mod.", 1)[1]: value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    top_k = None if args.topk <= 0 else args.topk
    soft_nms_sigma = None if args.soft_nms_sigma < 0 else args.soft_nms_sigma

    predictions: List[List[Dict[str, float]]] = []
    references: List[List[Dict[str, float]]] = []
    identities: List[Dict[str, Any]] = []

    data_iter = _dataset_iterator(
        pattern,
        batch_size=batch_size,
        shuffle_buf=0,
        sequence_length=sequence_length,
        text_tokens=text_tokens,
    )

    with torch.no_grad():
        for batch in data_iter:
            video = batch["video_feat"].to(device)
            text = batch["text_feat"].to(device)
            mask = batch["mask"].to(device)
            meta = batch["meta"]

            object_tensor = None
            if "object_feat" in batch:
                obj = batch["object_feat"].to(device)
                mask_obj = batch.get("object_mask")
                if mask_obj is not None:
                    obj = obj * batch["object_mask"].to(device).unsqueeze(-1)
                object_tensor = obj
            elif hasattr(model, "config") and getattr(model.config, "use_objects", False):
                object_dim = int(getattr(model.config, "object_dim", 0))
                if object_dim > 0:
                    object_tensor = torch.zeros(video.size(0), video.size(1), object_dim, device=device, dtype=video.dtype)

            use_objects = (
                object_tensor is not None
                and hasattr(model, "config")
                and getattr(model.config, "use_objects", False)
                and hasattr(model, "object_gate")
            )

            if use_objects:
                outputs = model(video, text, object_features=object_tensor)
            else:
                outputs = model(video, text)
            score_logits = outputs["scores"].squeeze(1)
            score_probs = torch.sigmoid(score_logits)
            bounds = outputs["bounds"]
            valid_mask = mask > 0

            for idx, info in enumerate(meta):
                valid_steps = valid_mask[idx].bool()
                step_scores = score_probs[idx][valid_steps].cpu()
                step_bounds = bounds[idx][:, valid_steps].cpu()
                duration = float(info.get("duration", 1.0))
                if duration <= 0:
                    duration = 1.0

                proposals: List[Dict[str, float]] = []
                for t in range(step_scores.shape[0]):
                    start_frac = (float(step_bounds[0, t]) + 1.0) / 2.0
                    end_frac = (float(step_bounds[1, t]) + 1.0) / 2.0
                    start_sec = max(0.0, min(duration, start_frac * duration))
                    end_sec = max(start_sec, min(duration, end_frac * duration))
                    proposals.append(
                        {
                            "start": start_sec,
                            "end": end_sec,
                            "score": float(step_scores[t]),
                        }
                    )

                if soft_nms_sigma is not None:
                    segments = postprocess.soft_nms_temporal(
                        proposals,
                        sigma=soft_nms_sigma,
                        iou_threshold=args.iou_threshold,
                        top_k=top_k,
                    )
                    proposals = [{"start": seg.start, "end": seg.end, "score": seg.score} for seg in segments]
                else:
                    proposals.sort(key=lambda item: item["score"], reverse=True)
                    if top_k is not None:
                        proposals = proposals[:top_k]

                predictions.append(_format_segments(proposals))
                identities.append(
                    {
                        "video_id": info.get("video_id"),
                        "qid": info.get("qid"),
                    }
                )
                if info.get("start") is not None and info.get("end") is not None:
                    references.append(
                        [
                            {
                                "start": float(info["start"]),
                                "end": float(info["end"]),
                            }
                        ]
                    )
                else:
                    references.append([])

    outputs = _prepare_outputs(config_path, args.split, output_dir=args.output_dir)

    prediction_records = []
    for meta, segments in zip(identities, predictions):
        prediction_records.append(
            {
                "video_id": meta["video_id"],
                "qid": meta["qid"],
                "segments": segments,
            }
        )

    thresholds = DEFAULT_THRESHOLDS
    metrics: Dict[str, Any] = {
        "split": args.split,
        "num_samples": len(prediction_records),
        "topk": top_k,
        "soft_nms_sigma": soft_nms_sigma,
    }

    if any(len(gt) for gt in references):
        recall_map = recall_iou.compute_recall_at_iou(
            predictions,
            references,
            thresholds=thresholds,
            max_predictions=top_k,
        )
        recall_k = recall_iou.recall_at_k(
            predictions,
            references,
            ks=DEFAULT_RECALL_K,
            iou_threshold=args.iou_threshold,
        )
        metrics["recall_iou"] = {f"{thr:.2f}": float(val) for thr, val in recall_map.items()}
        metrics["recall_at_k"] = {str(k): float(val) for k, val in recall_k.items()}
        metrics["map"] = float(
            recall_iou.average_precision_at_iou(
                predictions,
                references,
                iou_threshold=args.iou_threshold,
                max_predictions=top_k,
            )
        )

    outputs["predictions"].write_text(json.dumps(prediction_records, indent=2), encoding="utf-8")
    outputs["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    submission_payload = []
    for record in prediction_records:
        submission_payload.append(
            {
                "video_id": record["video_id"],
                "qid": record["qid"],
                "segments": record["segments"],
            }
        )
    outputs["submission"].write_text(json.dumps(submission_payload, indent=2), encoding="utf-8")

    print(f"Wrote predictions to {outputs['predictions']}")
    print(f"Wrote metrics to {outputs['metrics']}")
    print(f"Wrote submission to {outputs['submission']}")


if __name__ == "__main__":  # pragma: no cover
    main()
