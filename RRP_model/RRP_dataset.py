import os
import warnings
from pathlib import Path
from typing import Tuple

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

        if not isinstance(data_splits, dict) or self.split not in data_splits:
            raise KeyError(
                f"Split '{self.split}' is not defined in '{self.data_splits_path}'."
            )
        self.data_split = ["".join(x.split()) for x in data_splits[self.split]]
        if len(self.data_split) != len(set(self.data_split)):
            raise ValueError(
                f"Split '{self.split}' contains duplicate scenes in "
                f"'{self.data_splits_path}'."
            )
        self.data = self._load_data(self.data_folder, self.data_split)
        if not self.data:
            raise RuntimeError(f"No RRP samples were loaded for split '{self.split}'.")

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
        missing_scenes = []
        for scene in data_split:
            cur_dir = os.path.join(data_folder, scene)
            if not os.path.isdir(cur_dir):
                missing_scenes.append(scene)
                continue

            scene_format = self._scene_format(cur_dir)
            map_path = os.path.join(cur_dir, "map.png")
            pose_path = os.path.join(cur_dir, scene_format["pose_file"])
            depth_path = os.path.join(
                cur_dir,
                self.dataset_cfg.get("depth_file", "depth40.txt"),
            )
            rgb_dir = os.path.join(cur_dir, scene_format["rgb_dir"])
            required_paths = {
                "floorplan": map_path,
                "pose file": pose_path,
                "depth file": depth_path,
                "RGB directory": rgb_dir,
            }
            for path_type, path in required_paths.items():
                exists = (
                    os.path.isdir(path)
                    if path_type.endswith("directory")
                    else os.path.isfile(path)
                )
                if not exists:
                    raise FileNotFoundError(
                        f"Scene '{scene}' is missing its {path_type}: '{path}'."
                    )

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

            counts = (len(files), len(pose_data), len(ray_data))
            if len(set(counts)) != 1:
                raise ValueError(
                    f"Scene '{scene}' has mismatched sample counts: "
                    f"RGB images={counts[0]}, poses={counts[1]}, "
                    f"depth rows={counts[2]}."
                )
            if not files:
                raise ValueError(f"Scene '{scene}' contains no samples.")

            for n in range(len(files)):
                data.append(
                    {
                        "rgb_image": str(files[n]),
                        "floorplan_image": map_path,
                        "pose": pose_data[n],
                        "ray": ray_data[n],
                    }
                )
        if missing_scenes:
            examples = ", ".join(missing_scenes[:5])
            suffix = "" if len(missing_scenes) <= 5 else ", ..."
            warnings.warn(
                f"RRP split '{self.split}' lists {len(missing_scenes)} scene "
                f"directories that are unavailable under '{data_folder}'. "
                f"They were excluded: {examples}{suffix}",
                RuntimeWarning,
            )
        return data

    def _load_image(self, image_path, target_size):
        with open(image_path, "rb") as image_file:
            return img_path_to_data(image_file, target_size)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        data = self.data[i]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        with Image.open(data["rgb_image"]) as image:
            rgb_image = transform(image.convert("RGB"))

        pose = torch.tensor(data["pose"])
        ray = torch.tensor(data["ray"])

        with Image.open(data["floorplan_image"]) as image:
            w, h = image.size
        wh_tensor = torch.tensor([w, h], dtype=torch.float32)
        floorplan_img = self._load_image(
            data["floorplan_image"], self.floorplan_img_size
        )

        return (
            torch.as_tensor(rgb_image, dtype=torch.float32),
            torch.as_tensor(pose, dtype=torch.float32),
            torch.as_tensor(ray, dtype=torch.float32),
            torch.as_tensor(floorplan_img, dtype=torch.float32),
            torch.as_tensor(wh_tensor, dtype=torch.float32),
        )
