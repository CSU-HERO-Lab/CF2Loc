import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DisCo_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_splits_path: str,
        split: str,
        floorplan_img_size: Tuple[int, int],
        pose_aug_params: dict = None,
        dataset_cfg: dict = None,
    ):
        self.data_folder = data_folder
        self.data_splits_path = data_splits_path
        self.split = split
        self.floorplan_img_size = floorplan_img_size
        self.pose_aug_params = pose_aug_params if pose_aug_params else {"enable": False}
        self.dataset_cfg = dataset_cfg or {}
        self.dataset_type = self.dataset_cfg.get("dataset_type", "auto").lower()
        self.map_res = float(self.dataset_cfg.get("map_res", self._default_map_res()))

        with open(self.data_splits_path, "r", encoding="utf-8") as f:
            data_splits = yaml.safe_load(f)

        self.data_split = ["".join(x.split()) for x in data_splits[self.split]]
        self.data = self._load_data(self.data_folder, self.data_split)

    def _default_map_res(self):
        if self.dataset_type == "gibson" or "gibson" in self.data_folder.lower():
            return 0.01
        return 0.02

    def _scene_format(self, scene_dir):
        if self.dataset_type in ("s3d", "structured3d"):
            pose_in_meters = False
            pose_file = "poses_map.txt"
            rgb_dir = "imgs"
            map_res = 0.02
        elif self.dataset_type == "gibson":
            pose_in_meters = True
            pose_file = "poses.txt"
            rgb_dir = "rgb"
            map_res = 0.01
        elif os.path.exists(os.path.join(scene_dir, "poses_map.txt")):
            pose_in_meters = False
            pose_file = "poses_map.txt"
            rgb_dir = "imgs"
            map_res = 0.02
        else:
            pose_in_meters = True
            pose_file = "poses.txt"
            rgb_dir = "rgb"
            map_res = 0.01

        return {
            "pose_file": self.dataset_cfg.get("pose_file", pose_file),
            "rgb_dir": self.dataset_cfg.get("rgb_dir", rgb_dir),
            "pose_in_meters": self.dataset_cfg.get("pose_in_meters", pose_in_meters),
            "map_res": float(self.dataset_cfg.get("map_res", map_res)),
        }

    @staticmethod
    def _sort_key(path: Path):
        stem = path.stem
        if "-" in stem:
            major, minor = stem.split("-", 1)
            return (int(major), int(minor))
        return (int(stem), 0)

    @staticmethod
    def _convert_meter_poses_to_pixels(pose_data, map_path, map_res):
        with Image.open(map_path) as map_img:
            map_w, map_h = map_img.size

        for pose in pose_data:
            pose[0] = pose[0] / map_res + map_w / 2
            pose[1] = pose[1] / map_res + map_h / 2
        return pose_data

    def _load_data(self, data_folder, data_split):
        data = []
        for scene in data_split:
            cur_dir = os.path.join(data_folder, scene)
            if not os.path.exists(cur_dir):
                continue

            scene_format = self._scene_format(cur_dir)
            map_path = os.path.join(cur_dir, "map.png")
            pose_path = os.path.join(cur_dir, scene_format["pose_file"])
            depth_path = os.path.join(cur_dir, self.dataset_cfg.get("depth_file", "depth40.txt"))
            rgb_dir = os.path.join(cur_dir, scene_format["rgb_dir"])

            pose_data = [
                list(map(float, line.split()))[:3]
                for line in Path(pose_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if scene_format["pose_in_meters"]:
                pose_data = self._convert_meter_poses_to_pixels(
                    pose_data, map_path, scene_format["map_res"]
                )

            ray_data = [
                list(map(float, line.split()))
                for line in Path(depth_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            files = sorted(
                (
                    p
                    for p in Path(rgb_dir).iterdir()
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                ),
                key=self._sort_key,
            )

            sample_count = min(len(files), len(pose_data), len(ray_data))
            for n in range(sample_count):
                data.append(
                    {
                        "rgb_image": str(files[n]),
                        "floorplan_image": map_path,
                        "pose": pose_data[n],
                        "ray": ray_data[n],
                    }
                )
        return data

    def _load_floorplan(self, floorplan_path):
        try:
            with Image.open(floorplan_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.floorplan_img_size)
                return transforms.ToTensor()(img)
        except Exception as e:
            print(f"Failed to load floorplan {floorplan_path}: {e}")
            return torch.zeros((3, *self.floorplan_img_size))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        data = self.data[i]
        rgb_image = Image.open(data["rgb_image"])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        rgb_image = transform(rgb_image)

        pose = torch.tensor(data["pose"])
        ray = torch.tensor(data["ray"])

        w, h = Image.open(data["floorplan_image"]).size
        wh_tensor = torch.tensor([w, h], dtype=torch.float32)
        floorplan_img = self._load_floorplan(data["floorplan_image"])

        raw_map = cv2.imread(data["floorplan_image"], 0)
        if raw_map is None:
            local_map = torch.zeros((1, 128, 128), dtype=torch.float32)
            neg_local_map = torch.zeros((1, 128, 128), dtype=torch.float32)
            neg_pose = torch.zeros(3, dtype=torch.float32)
            print(f"Warning: Failed to load floorplan image {data['floorplan_image']}.")
        else:
            pose_aug = pose.numpy().copy()
            if self.split == "train" and self.pose_aug_params.get("enable", False):
                trans_range = self.pose_aug_params.get("trans_range", 25)
                rot_range = self.pose_aug_params.get("rot_range", 0.26)
                pose_aug[0] += np.random.uniform(-trans_range, trans_range)
                pose_aug[1] += np.random.uniform(-trans_range, trans_range)
                pose_aug[2] += np.random.uniform(-rot_range, rot_range)

            crop_size_meters = self.dataset_cfg.get("local_map_crop_size_meters", 5.0)
            local_map_np = self.crop_local_map(raw_map, pose_aug, crop_size_meters)
            local_map = torch.from_numpy(local_map_np).float().unsqueeze(0) / 255.0

            neg_pose_list = self.get_hard_negative_pose(pose.numpy())
            neg_pose = torch.tensor(neg_pose_list, dtype=torch.float32)
            neg_local_map_np = self.crop_local_map(
                raw_map, neg_pose.numpy(), crop_size_meters
            )
            neg_local_map = torch.from_numpy(neg_local_map_np).float().unsqueeze(0) / 255.0

        return (
            torch.as_tensor(rgb_image, dtype=torch.float32),
            torch.as_tensor(pose, dtype=torch.float32),
            torch.as_tensor(ray, dtype=torch.float32),
            torch.as_tensor(floorplan_img, dtype=torch.float32),
            torch.as_tensor(wh_tensor, dtype=torch.float32),
            local_map,
            neg_local_map,
            torch.as_tensor(neg_pose, dtype=torch.float32),
        )

    def get_hard_negative_pose(self, pose):
        x, y, theta = pose
        if np.random.rand() < 0.5:
            theta_new = theta + np.pi + np.random.uniform(-0.2, 0.2)
            return [x, y, theta_new]

        dist_m = np.random.uniform(1.5, 3.0)
        dist_px = dist_m / self.map_res
        angle = np.random.uniform(0, 2 * np.pi)
        x_new = x + dist_px * np.cos(angle)
        y_new = y + dist_px * np.sin(angle)
        theta_new = theta + np.random.uniform(-0.2, 0.2)
        return [x_new, y_new, theta_new]

    def crop_local_map(self, map_img, pose, crop_size_meters, output_size=128):
        x, y, theta = pose
        crop_size_px = int(crop_size_meters / self.map_res)

        h, w = map_img.shape
        pad = crop_size_px
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


if __name__ == "__main__":
    dataset = DisCo_Dataset(
        data_folder="datasets_s3d/Structured3D",
        data_splits_path="datasets_s3d/Structured3D/split.yaml",
        split="train",
        floorplan_img_size=(256, 256),
        pose_aug_params={"enable": True, "trans_range": 25, "rot_range": 0.26},
        dataset_cfg={"dataset_type": "s3d", "map_res": 0.02},
    )
    print(len(dataset))
