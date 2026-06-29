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
        self.hard_negative_mode = self._get_hard_negative_mode()
        self.floorplan_representation = self.dataset_cfg.get(
            "floorplan_representation",
            "rgb",
        )
        self.floorplan_wall_threshold = int(
            self.dataset_cfg.get("floorplan_wall_threshold", 250)
        )
        self.floorplan_fill_wall_dilation = int(
            self.dataset_cfg.get("floorplan_fill_wall_dilation", 1)
        )
        self.map_pose_rot_aug = self.dataset_cfg.get("map_pose_rot_aug", {})
        self.map_pose_rot_aug_enable = bool(
            self.map_pose_rot_aug.get("enable", False)
        )
        self.map_pose_rot_aug_p = float(self.map_pose_rot_aug.get("p", 0.0))
        self.map_pose_rot_aug_angles = self._parse_rotation_angles(
            self.map_pose_rot_aug.get("angles", [0, 90, 180, 270])
        )

        with open(self.data_splits_path, "r", encoding="utf-8") as f:
            data_splits = yaml.safe_load(f)

        self.data_split = ["".join(x.split()) for x in data_splits[self.split]]
        self.data = self._load_data(self.data_folder, self.data_split)

    @staticmethod
    def _parse_rotation_angles(angles):
        rotation_ks = []
        for angle in angles:
            angle = int(angle) % 360
            if angle % 90 != 0:
                raise ValueError(
                    "map_pose_rot_aug angles must be multiples of 90 degrees."
                )
            rotation_ks.append((angle // 90) % 4)
        if not rotation_ks:
            rotation_ks = [0]
        return rotation_ks

    def _sample_map_pose_rotation(self):
        if (
            self.split != "train"
            or not self.map_pose_rot_aug_enable
            or self.map_pose_rot_aug_p <= 0.0
            or np.random.rand() >= self.map_pose_rot_aug_p
        ):
            return 0
        return int(np.random.choice(self.map_pose_rot_aug_angles))

    @staticmethod
    def _rotate_array_90(array, rotation_k):
        rotation_k = int(rotation_k) % 4
        if rotation_k == 0:
            return array
        return np.ascontiguousarray(np.rot90(array, k=rotation_k))

    @staticmethod
    def _rotate_pose_wh_90(pose, width, height, rotation_k):
        rotation_k = int(rotation_k) % 4
        pose = np.asarray(pose, dtype=np.float32).copy()
        x, y, theta = pose
        if rotation_k == 1:
            pose[0] = y
            pose[1] = width - 1.0 - x
            pose[2] = theta - np.pi / 2.0
            new_width, new_height = height, width
        elif rotation_k == 2:
            pose[0] = width - 1.0 - x
            pose[1] = height - 1.0 - y
            pose[2] = theta + np.pi
            new_width, new_height = width, height
        elif rotation_k == 3:
            pose[0] = height - 1.0 - y
            pose[1] = x
            pose[2] = theta + np.pi / 2.0
            new_width, new_height = height, width
        else:
            new_width, new_height = width, height

        pose[0] = np.clip(pose[0], 0.0, max(float(new_width - 1), 0.0))
        pose[1] = np.clip(pose[1], 0.0, max(float(new_height - 1), 0.0))
        pose[2] = np.mod(pose[2], 2.0 * np.pi)
        return pose, int(new_width), int(new_height)

    def _default_map_res(self):
        if self.dataset_type in ("gibson", "zind") or "gibson" in self.data_folder.lower():
            return 0.01
        return 0.02

    def _get_hard_negative_mode(self):
        hard_negative_cfg = self.dataset_cfg.get("hard_negative", {})
        if isinstance(hard_negative_cfg, dict):
            mode = hard_negative_cfg.get("mode", None)
        else:
            mode = hard_negative_cfg

        mode = self.dataset_cfg.get("hard_negative_mode", mode)
        mode = (mode or "mixed").lower()
        aliases = {
            "pos": "position",
            "trans": "position",
            "translation": "position",
            "ori": "orientation",
            "rot": "orientation",
            "rotation": "orientation",
            "off": "none",
            "false": "none",
            "disable": "none",
            "disabled": "none",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("mixed", "position", "orientation", "none"):
            raise ValueError(
                "Unsupported hard negative mode "
                f"'{mode}'. Expected one of: mixed, position, orientation, none."
            )
        return mode

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
        elif self.dataset_type == "zind":
            # prepare_zind.py writes map-aligned pixel poses to a 1 cm/pixel map.
            pose_in_meters = False
            pose_file = "poses_map.txt"
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

    def _build_ternary_floorplan(self, gray_img):
        wall = (gray_img < self.floorplan_wall_threshold).astype(np.uint8)
        wall_for_fill = wall
        if self.floorplan_fill_wall_dilation > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            wall_for_fill = cv2.dilate(
                wall,
                kernel,
                iterations=self.floorplan_fill_wall_dilation,
            )

        free_candidate = (wall_for_fill == 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(free_candidate, connectivity=4)
        if num_labels <= 1:
            outside = np.zeros_like(free_candidate, dtype=bool)
        else:
            border_labels = np.unique(
                np.concatenate(
                    [
                        labels[0, :],
                        labels[-1, :],
                        labels[:, 0],
                        labels[:, -1],
                    ]
                )
            )
            border_labels = border_labels[border_labels != 0]
            outside = np.isin(labels, border_labels)

        indoor_free = (free_candidate > 0) & ~outside
        ternary = np.zeros_like(gray_img, dtype=np.float32)
        ternary[wall_for_fill > 0] = 0.5
        ternary[indoor_free] = 1.0
        return ternary

    def _load_floorplan(self, floorplan_path, rotation_k=0):
        try:
            with Image.open(floorplan_path) as img:
                if self.floorplan_representation == "gray_ternary":
                    img = img.convert("L")
                    gray = np.asarray(img, dtype=np.uint8)
                    gray = self._rotate_array_90(gray, rotation_k)
                    img = Image.fromarray(gray, mode="L")
                    img = img.resize(self.floorplan_img_size)
                    gray = np.asarray(img, dtype=np.uint8)
                    gray_float = gray.astype(np.float32) / 255.0
                    ternary = self._build_ternary_floorplan(gray)
                    stacked = np.stack([gray_float, ternary], axis=0)
                    return torch.from_numpy(stacked).float()

                img = img.convert("RGB")
                rgb = np.asarray(img, dtype=np.uint8)
                rgb = self._rotate_array_90(rgb, rotation_k)
                img = Image.fromarray(rgb, mode="RGB")
                img = img.resize(self.floorplan_img_size)
                return transforms.ToTensor()(img)
        except Exception as e:
            print(f"Failed to load floorplan {floorplan_path}: {e}")
            if self.floorplan_representation == "gray_ternary":
                return torch.zeros((2, *self.floorplan_img_size))
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
        rotation_k = self._sample_map_pose_rotation()
        if rotation_k != 0:
            pose_np, w, h = self._rotate_pose_wh_90(
                pose.numpy(),
                w,
                h,
                rotation_k,
            )
            pose = torch.from_numpy(pose_np)
        wh_tensor = torch.tensor([w, h], dtype=torch.float32)
        floorplan_img = self._load_floorplan(data["floorplan_image"], rotation_k)

        raw_map = cv2.imread(data["floorplan_image"], 0)
        if raw_map is None:
            local_map = torch.zeros((1, 128, 128), dtype=torch.float32)
            neg_local_map = torch.zeros((1, 128, 128), dtype=torch.float32)
            neg_pose = torch.zeros(3, dtype=torch.float32)
            print(f"Warning: Failed to load floorplan image {data['floorplan_image']}.")
        else:
            raw_map = self._rotate_array_90(raw_map, rotation_k)
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

            if self.hard_negative_mode == "none":
                neg_pose = torch.zeros(3, dtype=torch.float32)
                neg_local_map = torch.zeros((1, 128, 128), dtype=torch.float32)
            else:
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
        if self.hard_negative_mode == "orientation" or (
            self.hard_negative_mode == "mixed" and np.random.rand() < 0.5
        ):
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
