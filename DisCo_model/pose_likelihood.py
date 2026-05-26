from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch


def wrap_angle_2pi(angle: float) -> float:
    return float(np.mod(angle, 2.0 * np.pi))


def crop_local_map(
    map_img,
    x: float,
    y: float,
    theta: float,
    crop_size_meters: float,
    res: float = 0.02,
    output_size: int = 128,
):
    x = float(x)
    y = float(y)
    crop_size_px = max(1, int(crop_size_meters / res))
    pad = crop_size_px

    if torch.is_tensor(map_img):
        map_img = map_img.detach().cpu().numpy()
    if len(map_img.shape) == 3:
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

    map_padded = cv2.copyMakeBorder(
        map_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
    )

    center = (x + pad, y + pad)
    angle_deg = np.degrees(theta)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg + 90, 1.0)
    rot_matrix[0, 2] += (crop_size_px / 2.0) - center[0]
    rot_matrix[1, 2] += (crop_size_px / 2.0) - center[1]

    local_map = cv2.warpAffine(
        map_padded,
        rot_matrix,
        (crop_size_px, crop_size_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    if crop_size_px != output_size:
        local_map = cv2.resize(
            local_map, (output_size, output_size), interpolation=cv2.INTER_AREA
        )
    return local_map


def _symmetric_offsets(radius: float) -> Iterable[float]:
    if radius <= 0:
        return (0.0,)
    radius = float(radius)
    return (0.0, -radius, radius)


def generate_se2_neighborhood(
    base_poses_desdf,
    translation_radius_m: float,
    theta_radius_deg: float,
    meters_per_desdf_cell: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a small structured SE(2) grid around each DESDF-frame pose."""
    base_poses = np.asarray(base_poses_desdf, dtype=np.float32).reshape(-1, 3)
    xy_offsets_m = list(_symmetric_offsets(float(translation_radius_m)))
    theta_offsets_rad = [
        np.deg2rad(delta_deg) for delta_deg in _symmetric_offsets(float(theta_radius_deg))
    ]

    poses = []
    group_indices = []
    for group_idx, base_pose in enumerate(base_poses):
        base_x, base_y, base_theta = base_pose
        for dy_m in xy_offsets_m:
            for dx_m in xy_offsets_m:
                for dtheta in theta_offsets_rad:
                    poses.append(
                        [
                            base_x + dx_m / meters_per_desdf_cell,
                            base_y + dy_m / meters_per_desdf_cell,
                            wrap_angle_2pi(base_theta + dtheta),
                        ]
                    )
                    group_indices.append(group_idx)

    return np.asarray(poses, dtype=np.float32), np.asarray(group_indices, dtype=np.int64)


def desdf_poses_to_map_pixels(
    poses_desdf,
    desdf_origin: Tuple[float, float],
    desdf_stride: float,
) -> np.ndarray:
    poses_desdf = np.asarray(poses_desdf, dtype=np.float32).reshape(-1, 3)
    origin_l, origin_t = desdf_origin
    poses_map = poses_desdf.copy()
    poses_map[:, 0] = poses_desdf[:, 0] * desdf_stride + origin_l
    poses_map[:, 1] = poses_desdf[:, 1] * desdf_stride + origin_t
    return poses_map


def build_local_map_batch(
    map_img,
    poses_desdf,
    desdf_origin: Tuple[float, float],
    desdf_stride: float,
    map_res: float,
    crop_size_meters: float,
    output_size: int = 128,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    poses_map = desdf_poses_to_map_pixels(poses_desdf, desdf_origin, desdf_stride)
    local_maps = []
    for x_map, y_map, theta in poses_map:
        local_map = crop_local_map(
            map_img,
            x_map,
            y_map,
            theta,
            crop_size_meters=crop_size_meters,
            res=map_res,
            output_size=output_size,
        )
        local_maps.append(torch.from_numpy(local_map).float().unsqueeze(0) / 255.0)

    batch = torch.stack(local_maps)
    if device is not None:
        batch = batch.to(device)
    return batch


def _select_group(logits: torch.Tensor, group_indices: torch.Tensor, reduce: str) -> int:
    unique_groups = torch.unique(group_indices, sorted=True)
    group_scores = []
    for group_idx in unique_groups:
        group_logits = logits[group_indices == group_idx]
        if reduce == "argmax":
            group_scores.append(torch.max(group_logits))
        else:
            group_scores.append(torch.logsumexp(group_logits, dim=0))
    group_scores = torch.stack(group_scores)
    return int(unique_groups[torch.argmax(group_scores)].item())


def _reduce_pose(poses: torch.Tensor, logits: torch.Tensor, reduce: str) -> torch.Tensor:
    if reduce == "argmax":
        return poses[torch.argmax(logits)]

    weights = torch.softmax(logits, dim=0)
    xy = torch.sum(poses[:, :2] * weights.unsqueeze(-1), dim=0)
    sin_theta = torch.sum(torch.sin(poses[:, 2]) * weights)
    cos_theta = torch.sum(torch.cos(poses[:, 2]) * weights)
    theta = torch.remainder(torch.atan2(sin_theta, cos_theta), 2.0 * torch.pi)
    return torch.cat([xy, theta.view(1)])


def refine_pose_likelihood(
    model,
    img_tokens,
    map_img,
    base_poses_desdf,
    desdf_origin: Tuple[float, float],
    desdf_stride: float,
    map_res: float,
    crop_size_meters: float,
    translation_radius_m: float = 0.1,
    theta_radius_deg: float = 5.0,
    base_scores=None,
    score_scale: float = 1.0,
    reduce: str = "softargmax",
    output_size: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Score a structured SE(2) neighborhood with DisCo and return a refined pose.

    The helper keeps mode selection discrete when multiple base poses are supplied,
    then applies the requested reducer inside the selected local neighborhood.
    """
    if reduce not in ("softargmax", "argmax"):
        raise ValueError("reduce must be 'softargmax' or 'argmax'.")

    meters_per_desdf_cell = float(desdf_stride) * float(map_res)
    grid_poses, group_indices_np = generate_se2_neighborhood(
        base_poses_desdf,
        translation_radius_m=translation_radius_m,
        theta_radius_deg=theta_radius_deg,
        meters_per_desdf_cell=meters_per_desdf_cell,
    )
    if grid_poses.size == 0:
        raise ValueError("base_poses_desdf must contain at least one pose.")

    if device is None:
        device = next(model.parameters()).device

    candidate_maps = build_local_map_batch(
        map_img,
        grid_poses,
        desdf_origin=desdf_origin,
        desdf_stride=desdf_stride,
        map_res=map_res,
        crop_size_meters=crop_size_meters,
        output_size=output_size,
        device=device,
    )

    sim_scores = model.score_candidates(img_tokens, candidate_maps)
    logits = sim_scores * float(score_scale)
    group_indices = torch.from_numpy(group_indices_np).to(device=device, dtype=torch.long)

    if base_scores is not None:
        base_scores_t = torch.as_tensor(base_scores, dtype=logits.dtype, device=device)
        base_log_prior = torch.log(torch.clamp(base_scores_t, min=1e-12))
        logits = logits + base_log_prior[group_indices]

    selected_group = _select_group(logits, group_indices, reduce=reduce)
    selected_mask = group_indices == selected_group
    selected_logits = logits[selected_mask]
    selected_poses = torch.as_tensor(grid_poses, dtype=logits.dtype, device=device)[
        selected_mask
    ]
    refined_pose = _reduce_pose(selected_poses, selected_logits, reduce=reduce)

    return {
        "pose_desdf": refined_pose.detach().cpu().numpy(),
        "selected_base": selected_group,
        "num_grid_poses": int(grid_poses.shape[0]),
        "grid_poses_desdf": grid_poses,
        "logits": logits.detach().cpu(),
        "sim_scores": sim_scores.detach().cpu(),
        "group_indices": group_indices_np,
    }
