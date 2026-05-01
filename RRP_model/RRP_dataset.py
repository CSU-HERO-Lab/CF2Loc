import os
from pathlib import Path
from typing import Tuple

import cv2
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from DisCo_model.data_utils import img_path_to_data


class RRP_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_splits_path: str,
        split: str,
        rgb_image_size: Tuple[int, int],
        floorplan_img_size: Tuple[int, int],
        dataset_cfg: dict = None,
    ):
        self.data_folder = data_folder
        self.data_splits_path = data_splits_path
        self.split = split
        self.rgb_image_size = rgb_image_size
        self.floorplan_img_size = floorplan_img_size
        self.dataset_cfg = dataset_cfg or {}
        self.dataset_type = self.dataset_cfg.get("dataset_type", "auto").lower()

        with open(self.data_splits_path, "r", encoding="utf-8") as f:
            data_splits = yaml.safe_load(f)

        self.data_split = ["".join(x.split()) for x in data_splits[self.split]]
        self.data = self._load_data(self.data_folder, self.data_split)

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
                print(f"Warning: {cur_dir} does not exist, skipping.")
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

    def _load_image(self, image_path, target_size):
        try:
            with open(image_path, "rb") as f:
                return img_path_to_data(f, target_size)
        except TypeError:
            print(f"Failed to load image {image_path}")
            return torch.zeros((3, target_size[1], target_size[0]), dtype=torch.float32)

    def _load_floorplan(self, floorplan_path):
        floorplan_img = cv2.imread(floorplan_path, 0)
        return floorplan_img

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
        floorplan_img = self._load_image(data["floorplan_image"], self.floorplan_img_size)

        return (
            torch.as_tensor(rgb_image, dtype=torch.float32),
            torch.as_tensor(pose, dtype=torch.float32),
            torch.as_tensor(ray, dtype=torch.float32),
            torch.as_tensor(floorplan_img, dtype=torch.float32),
            torch.as_tensor(wh_tensor, dtype=torch.float32),
        )


if __name__ == "__main__":
    dataset = RRP_Dataset(
        data_folder="datasets_s3d/Structured3D",
        data_splits_path="datasets_s3d/Structured3D/split.yaml",
        split="train",
        rgb_image_size=(640, 480),
        floorplan_img_size=(256, 256),
    )
    print(len(dataset))
