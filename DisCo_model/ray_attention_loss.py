import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def infer_square_grid(num_tokens: int) -> Tuple[int, int]:
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size != num_tokens:
        raise ValueError(
            f"Cannot infer a square map-token grid from {num_tokens} tokens. "
            "Pass feat_h and feat_w explicitly."
        )
    return grid_size, grid_size


def build_ray_attention_targets(
    ray: torch.Tensor,
    local_map: Optional[torch.Tensor],
    num_image_tokens: int,
    num_map_tokens: int,
    image_token_grid: Tuple[int, int] = (6, 40),
    feat_h: Optional[int] = None,
    feat_w: Optional[int] = None,
    crop_size_meters: float = 5.0,
    fov_degrees: float = 80.0,
    endpoint_sigma: float = 0.08,
    corridor_width: float = 0.045,
    corridor_weight: float = 0.35,
    wall_weight: float = 0.25,
    eps: float = 1.0e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build weak image-token to map-token attention targets from depth/ray bins.

    The local crop is treated as a metric square centered at the camera, with
    the camera looking toward the top of the crop. Each ray bin supervises the
    image tokens in the corresponding image column with an endpoint + corridor
    distribution over the map feature grid.
    """
    if ray.dim() > 2:
        ray = ray.view(ray.shape[0], -1)
    ray = ray.float()

    batch_size, source_bins = ray.shape
    img_h, img_w = image_token_grid
    grid_image_tokens = img_h * img_w
    if feat_h is None or feat_w is None:
        feat_h, feat_w = infer_square_grid(num_map_tokens)
    if feat_h * feat_w != num_map_tokens:
        raise ValueError(
            f"Map grid shape ({feat_h}, {feat_w}) does not match "
            f"{num_map_tokens} attention tokens."
        )

    device = ray.device
    dtype = ray.dtype

    if source_bins != img_w:
        ray = F.interpolate(
            ray.unsqueeze(1), size=img_w, mode="linear", align_corners=False
        ).squeeze(1)

    ray_valid = torch.isfinite(ray) & (ray > 0)
    ray = torch.nan_to_num(ray, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    ray_norm = (ray / max(float(crop_size_meters), eps)).clamp(0.0, 1.5)

    angles = torch.linspace(
        -0.5 * float(fov_degrees),
        0.5 * float(fov_degrees),
        img_w,
        device=device,
        dtype=dtype,
    )
    angles = torch.deg2rad(angles)

    center = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
    endpoint_x = center[0] + ray_norm * torch.sin(angles).unsqueeze(0)
    endpoint_y = center[1] - ray_norm * torch.cos(angles).unsqueeze(0)
    endpoints = torch.stack((endpoint_x, endpoint_y), dim=-1).clamp(0.0, 1.0)

    pos_x = torch.linspace(0.0, 1.0, feat_w, device=device, dtype=dtype)
    pos_y = torch.linspace(0.0, 1.0, feat_h, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, num_map_tokens, 2)

    endpoint_delta = grid - endpoints.unsqueeze(2)
    endpoint_score = torch.exp(
        -endpoint_delta.square().sum(dim=-1) / (2.0 * endpoint_sigma * endpoint_sigma)
    )

    segment = (endpoints - center).unsqueeze(2)
    point_delta = grid - center
    segment_len_sq = segment.square().sum(dim=-1).clamp_min(eps)
    projection = (point_delta * segment).sum(dim=-1) / segment_len_sq
    on_segment = (projection >= 0.0) & (projection <= 1.0)
    projection = projection.clamp(0.0, 1.0).unsqueeze(-1)
    nearest = center + projection * segment
    corridor_dist_sq = (grid - nearest).square().sum(dim=-1)
    corridor_score = torch.exp(
        -corridor_dist_sq / (2.0 * corridor_width * corridor_width)
    )
    corridor_score = corridor_score * on_segment.to(dtype)

    target_by_ray = endpoint_score + float(corridor_weight) * corridor_score

    if local_map is not None and wall_weight > 0:
        wall_map = 1.0 - local_map.float().mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        wall_map = F.interpolate(
            wall_map, size=(feat_h, feat_w), mode="bilinear", align_corners=False
        )
        wall_tokens = wall_map.flatten(1).unsqueeze(1)
        target_by_ray = target_by_ray * (1.0 + float(wall_weight) * wall_tokens)

    uniform = torch.full_like(target_by_ray, 1.0 / num_map_tokens)
    target_by_ray = torch.where(ray_valid.unsqueeze(-1), target_by_ray, uniform)
    target_by_ray = target_by_ray / target_by_ray.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)

    token_targets = torch.zeros(
        batch_size, num_image_tokens, num_map_tokens, device=device, dtype=dtype
    )
    token_valid = torch.zeros(
        batch_size, num_image_tokens, device=device, dtype=torch.bool
    )

    usable_tokens = min(num_image_tokens, grid_image_tokens)
    start = num_image_tokens - usable_tokens
    grid_indices = torch.arange(usable_tokens, device=device)
    cols = (grid_indices % img_w).long()

    token_targets[:, start:, :] = target_by_ray[:, cols, :]
    token_valid[:, start:] = ray_valid[:, cols]

    return token_targets, token_valid


def ray_attention_kl_loss(
    attention: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if attention.shape != target.shape:
        raise ValueError(
            f"Attention shape {tuple(attention.shape)} does not match "
            f"target shape {tuple(target.shape)}."
        )

    attention = attention.clamp_min(eps)
    target = target.clamp_min(eps)
    per_token = F.kl_div(attention.log(), target, reduction="none").sum(dim=-1)
    per_token = per_token * valid_mask.to(per_token.dtype)
    denom = valid_mask.to(per_token.dtype).sum().clamp_min(1.0)
    return per_token.sum() / denom
