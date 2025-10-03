"""OGSF model implementations combining gating, multi-scale heads, and ASL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import fpn1d, gate
from .losses import asl, focal, iou1d
from .mvp import MVPConfig, MVPModel

__all__ = ["OGSFConfig", "OGSFModel"]


@dataclass
class OGSFConfig(MVPConfig):
    """Configuration for the OGSF model."""

    use_objects: bool = True
    use_multiscale: bool = True
    use_asl: bool = True
    object_dim: int = 0
    fpn_levels: int = 3


class OGSFModel(nn.Module):
    """OGSF model wiring MVP backbone with gating and FPN branches."""

    def __init__(self, config: OGSFConfig) -> None:
        super().__init__()
        self.config = config

        self.mvp = MVPModel(MVPConfig(**{k: getattr(config, k) for k in ("d_v", "d_t", "hidden", "heads", "num_layers", "dropout", "use_flash_attn")}))

        self.object_gate: Optional[gate.ObjectGate]
        if config.use_objects:
            if config.object_dim <= 0:
                raise ValueError("object_dim must be positive when use_objects=True")
            self.object_gate = gate.ObjectGate(gate.ObjectGateConfig(input_dim=config.object_dim, hidden_dim=config.hidden))
        else:
            self.object_gate = None

        if config.use_multiscale:
            self.fpn = fpn1d.TemporalFPN(
                fpn1d.TemporalFPNConfig(
                    in_channels=config.hidden,
                    out_channels=config.hidden,
                    num_levels=config.fpn_levels,
                )
            )
        else:
            self.fpn = None

    def forward(
        self,
        video: Tensor,
        text: Tensor,
        *,
        object_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor | List[Dict[str, Tensor]]]:
        fused = self.mvp.encode(video, text)
        gates = None

        if self.object_gate is not None:
            if object_features is None:
                raise ValueError("object_features required when use_objects=True")
            fused, gates = self.object_gate(fused, object_features)

        multiscale_outputs: List[Dict[str, Tensor]] = []
        if self.fpn is not None:
            for level_features in self.fpn(fused):
                scores, bounds = self.mvp.head(level_features)
                multiscale_outputs.append({"scores": scores, "bounds": bounds, "features": level_features})

            main_scores = multiscale_outputs[0]["scores"]
            main_bounds = multiscale_outputs[0]["bounds"]
        else:
            main_scores, main_bounds = self.mvp.head(fused)

        result: Dict[str, Tensor | List[Dict[str, Tensor]]] = {
            "scores": main_scores,
            "bounds": main_bounds,
            "features": fused,
        }

        if gates is not None:
            result["gates"] = gates
        if multiscale_outputs:
            result["multiscale"] = multiscale_outputs

        return result

    def compute_loss(
        self,
        predictions: Dict[str, Tensor | List[Dict[str, Tensor]]],
        targets: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute loss respecting configuration flags."""

        score_logits = predictions["scores"].squeeze(1)
        score_targets = targets["scores"].to(device=score_logits.device, dtype=score_logits.dtype)

        if self.config.use_asl and "sensitivity" in targets:
            score_loss = asl.asl_loss(score_logits, score_targets, targets["sensitivity"].to(score_logits.device, score_logits.dtype))
        else:
            score_loss = focal.focal_loss(score_logits, score_targets)

        bounds_pred = predictions["bounds"]
        bounds_target = targets["bounds"].to(device=bounds_pred.device, dtype=bounds_pred.dtype)
        iou_component = iou1d.iou_loss(bounds_pred.transpose(1, 2), bounds_target.transpose(1, 2), reduction="mean")
        bound_loss = iou_component + F.l1_loss(bounds_pred, bounds_target)

        components: Dict[str, Tensor] = {"score": score_loss, "bounds": bound_loss}

        if self.config.use_multiscale and "multiscale" in predictions:
            multiscale_loss = bounds_pred.new_tensor(0.0)
            for branch in predictions["multiscale"][1:]:
                scores_branch = branch["scores"].squeeze(1)
                bounds_branch = branch["bounds"]

                scaled_scores = self._resize(scores_branch, score_targets.shape[-1])
                scaled_bounds = self._resize(bounds_branch, bounds_target.shape[-1])

                if self.config.use_asl and "sensitivity" in targets:
                    branch_score = asl.asl_loss(scaled_scores, score_targets, targets["sensitivity"].to(score_logits.device, score_logits.dtype))
                else:
                    branch_score = focal.focal_loss(scaled_scores, score_targets)

                branch_bounds = iou1d.iou_loss(
                    scaled_bounds.transpose(1, 2),
                    bounds_target.transpose(1, 2),
                    reduction="mean",
                ) + F.l1_loss(scaled_bounds, bounds_target)

                multiscale_loss = multiscale_loss + 0.5 * (branch_score + branch_bounds)

            components["multiscale"] = multiscale_loss

        total = sum(components.values())
        return total, components

    @staticmethod
    def _resize(tensor: Tensor, length: int) -> Tensor:
        if tensor.shape[-1] == length:
            return tensor
        reshaped = tensor
        if tensor.ndim == 2:
            reshaped = tensor.unsqueeze(1)
        resized = F.interpolate(reshaped, size=length, mode="linear", align_corners=False)
        if tensor.ndim == 2:
            return resized.squeeze(1)
        return resized
