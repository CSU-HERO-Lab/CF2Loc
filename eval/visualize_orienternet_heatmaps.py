import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import yaml

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.orienternet_likelihood import OrienterNetLikelihoodModel


def denormalize_image(image):
    image = image.detach().cpu().float().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = np.clip(image * std + mean, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def to_uint8_heatmap(values):
    values = values.astype(np.float32)
    values = values - values.min()
    values = values / (values.max() + 1e-8)
    return (values * 255).astype(np.uint8)


def draw_pose(canvas, x, y, theta, color, label):
    x_i, y_i = int(round(x)), int(round(y))
    cv2.circle(canvas, (x_i, y_i), 5, color, -1, lineType=cv2.LINE_AA)
    arrow_len = 18
    end = (
        int(round(x_i + arrow_len * np.cos(theta))),
        int(round(y_i + arrow_len * np.sin(theta))),
    )
    cv2.arrowedLine(canvas, (x_i, y_i), end, color, 2, tipLength=0.35)
    cv2.putText(
        canvas,
        label,
        (x_i + 7, y_i - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def make_panel(obs_img, floorplan, heatmap, best_slice, gt_pose, pred_pose, wh, out_path):
    floor_rgb = floorplan.detach().cpu().float().numpy().transpose(1, 2, 0)
    floor_rgb = np.clip(floor_rgb * 255, 0, 255).astype(np.uint8)
    floor_rgb = cv2.cvtColor(floor_rgb, cv2.COLOR_RGB2BGR)
    obs_rgb = cv2.cvtColor(denormalize_image(obs_img), cv2.COLOR_RGB2BGR)

    height, width = floor_rgb.shape[:2]
    heat_u8 = to_uint8_heatmap(heatmap)
    heat_u8 = cv2.resize(heat_u8, (width, height), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(floor_rgb, 0.55, heat_color, 0.45, 0.0)

    slice_u8 = to_uint8_heatmap(best_slice)
    slice_u8 = cv2.resize(slice_u8, (width, height), interpolation=cv2.INTER_CUBIC)
    slice_color = cv2.applyColorMap(slice_u8, cv2.COLORMAP_TURBO)
    slice_overlay = cv2.addWeighted(floor_rgb, 0.55, slice_color, 0.45, 0.0)

    scale_x = width / float(wh[0])
    scale_y = height / float(wh[1])
    gt_x, gt_y, gt_theta = gt_pose
    pred_x, pred_y, pred_theta = pred_pose
    for canvas in (overlay, slice_overlay):
        draw_pose(canvas, gt_x * scale_x, gt_y * scale_y, gt_theta, (0, 255, 0), "GT")
        draw_pose(
            canvas,
            pred_x * scale_x,
            pred_y * scale_y,
            pred_theta,
            (0, 0, 255),
            "PRED",
        )

    obs_panel = cv2.resize(obs_rgb, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(obs_panel, "observation", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(overlay, "max_theta likelihood", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(slice_overlay, "pred_theta likelihood", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    panel = np.concatenate([obs_panel, overlay, slice_overlay], axis=1)
    cv2.imwrite(out_path, panel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="OrienterNet_FLoc.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--indices", default="0,8,32,128,512,1024")
    parser.add_argument("--out_dir", default="logs/orienternet_runs/method1_orienternet_dense/heatmaps")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split = args.split or config["datasets"].get("val_split", "val")
    dataset = DisCo_Dataset(
        data_folder=config["datasets"]["data_folder"],
        data_splits_path=config["datasets"]["data_splits"],
        split=split,
        floorplan_img_size=tuple(config["datasets"]["floorplan_img_size"]),
        pose_aug_params=None,
        dataset_cfg=config["datasets"],
    )

    model = OrienterNetLikelihoodModel.load_from_checkpoint(
        args.ckpt, config=config, map_location=device
    )
    model.to(device)
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    requested = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    saved = []
    with torch.no_grad():
        for sample_idx in requested:
            if sample_idx >= len(dataset):
                continue
            obs_img, pose, _ray, floorplan_img, wh, *_ = dataset[sample_idx]
            logits = model(
                obs_img.unsqueeze(0).to(device),
                floorplan_img.unsqueeze(0).to(device),
            )
            prob = torch.softmax(logits.flatten(1), dim=1).view_as(logits)[0]
            pred_x, pred_y, pred_theta = model.predict_pose(logits, wh.unsqueeze(0).to(device))
            pred_theta_idx = int(torch.argmax(logits.flatten(1), dim=1)[0] // (logits.shape[-2] * logits.shape[-1]))

            heatmap = prob.max(dim=0).values.detach().cpu().numpy()
            best_slice = prob[pred_theta_idx].detach().cpu().numpy()
            pred_pose = np.array(
                [
                    pred_x.item(),
                    pred_y.item(),
                    pred_theta.item(),
                ],
                dtype=np.float32,
            )
            gt_pose = pose.detach().cpu().numpy()
            wh_np = wh.detach().cpu().numpy()

            out_path = os.path.join(args.out_dir, f"{split}_{sample_idx:06d}.png")
            make_panel(
                obs_img,
                floorplan_img,
                heatmap,
                best_slice,
                gt_pose,
                pred_pose,
                wh_np,
                out_path,
            )
            saved.append(out_path)

    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
