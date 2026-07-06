import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from DisCo_model.image_patch_encoder import ImagePatchEncoder


class DenseFloorplanEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        output_stride: int = 8,
        input_mode: str = "gray",
        context_blocks: int = 1,
    ):
        super().__init__()
        if output_stride not in (2, 4, 8):
            raise ValueError("output_stride must be one of {2, 4, 8}.")
        if input_mode not in ("gray", "gray_edges", "rgb", "semantic_onehot"):
            raise ValueError(
                "input_mode must be one of "
                "{'gray', 'gray_edges', 'rgb', 'semantic_onehot'}."
            )

        self.input_mode = input_mode
        self.input_channels = {
            "gray": 1,
            "gray_edges": 3,
            "rgb": 3,
            "semantic_onehot": 5,
        }[input_mode]

        stride_plan = {
            2: (2, 1, 1),
            4: (2, 2, 1),
            8: (2, 2, 2),
        }[output_stride]

        layers = []
        in_channels = self.input_channels
        for out_channels, stride in zip((32, 64, feature_dim), stride_plan):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=5 if in_channels == 1 else 3,
                        stride=stride,
                        padding=2 if in_channels == 1 else 1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        layers.extend(
            [
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.GELU(),
            ]
        )
        self.stem = nn.Sequential(*layers)
        self.context = nn.Sequential(
            *[ResidualConvBlock(feature_dim) for _ in range(max(0, context_blocks))]
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, floorplan_img: torch.Tensor) -> torch.Tensor:
        if (
            self.input_mode not in ("rgb", "semantic_onehot")
            and floorplan_img.shape[1] == 3
        ):
            floorplan_img = floorplan_img.mean(dim=1, keepdim=True)
        floorplan_img = floorplan_img.float()
        if self.input_mode == "gray_edges":
            grad_x = F.conv2d(floorplan_img, self.sobel_x, padding=1)
            grad_y = F.conv2d(floorplan_img, self.sobel_y, padding=1)
            floorplan_img = torch.cat([floorplan_img, grad_x, grad_y], dim=1)
        return self.context(self.stem(floorplan_img))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ResNetFloorplanEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        input_mode: str = "gray_edges",
        context_blocks: int = 1,
        pretrained: bool = False,
    ):
        super().__init__()
        if input_mode not in (
            "gray",
            "gray_edges",
            "gray_ternary",
            "rgb",
            "semantic_onehot",
        ):
            raise ValueError(
                "input_mode must be one of "
                "{'gray', 'gray_edges', 'gray_ternary', 'rgb', 'semantic_onehot'}."
            )

        self.input_mode = input_mode
        input_channels = {
            "gray": 1,
            "gray_edges": 3,
            "gray_ternary": 2,
            "rgb": 3,
            "semantic_onehot": 5,
        }[input_mode]
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        if input_channels != 3:
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    backbone.conv1.weight.copy_(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.proj = nn.Conv2d(128, feature_dim, kernel_size=1, bias=False)
        self.proj_norm = nn.GroupNorm(min(8, feature_dim), feature_dim)
        self.context = nn.Sequential(
            *[ResidualConvBlock(feature_dim) for _ in range(max(0, context_blocks))]
        )

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, floorplan_img: torch.Tensor) -> torch.Tensor:
        if (
            self.input_mode not in ("gray_ternary", "rgb", "semantic_onehot")
            and floorplan_img.shape[1] == 3
        ):
            floorplan_img = floorplan_img.mean(dim=1, keepdim=True)
        floorplan_img = floorplan_img.float()
        if self.input_mode == "gray_edges":
            grad_x = F.conv2d(floorplan_img, self.sobel_x, padding=1)
            grad_y = F.conv2d(floorplan_img, self.sobel_y, padding=1)
            floorplan_img = torch.cat([floorplan_img, grad_x, grad_y], dim=1)

        features = self.relu(self.bn1(self.conv1(floorplan_img)))
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = F.gelu(self.proj_norm(self.proj(features)))
        return self.context(features)


class ObservationKernelEncoder(nn.Module):
    def __init__(
        self,
        image_encoder_name: str,
        dptv2_ckpt_path: str,
        image_feature_dim: int,
        map_feature_dim: int,
        theta_bins: int,
        kernel_size: int,
        image_token_grid: Tuple[int, int],
        num_heads: int,
        image_self_attn_layers: int,
        freeze_image_backbone: bool,
    ):
        super().__init__()
        self.theta_bins = theta_bins
        self.kernel_size = kernel_size
        self.image_token_grid = tuple(image_token_grid)

        self.image_encoder = ImagePatchEncoder(
            encoder=image_encoder_name,
            feature_dim=image_feature_dim,
            target_size=self.image_token_grid,
            checkpoint_path=dptv2_ckpt_path,
            freeze_backbone=freeze_image_backbone,
            use_cls_token=False,
        )
        self.token_norm = nn.LayerNorm(image_feature_dim)
        self.token_mixer = self._build_token_mixer(
            image_feature_dim, num_heads, image_self_attn_layers
        )
        self.obs_conv = nn.Sequential(
            nn.Conv2d(image_feature_dim, image_feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(image_feature_dim, image_feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.kernel_head = nn.Conv2d(
            image_feature_dim,
            theta_bins * map_feature_dim,
            kernel_size=1,
        )

    @staticmethod
    def _build_token_mixer(feature_dim: int, num_heads: int, num_layers: int):
        if num_layers <= 0:
            return nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(feature_dim),
        )

    def forward(self, obs_img: torch.Tensor) -> torch.Tensor:
        tokens = self.image_encoder(obs_img)
        tokens = self.token_norm(tokens)
        tokens = self.token_mixer(tokens)

        bsz, _, channels = tokens.shape
        feat_h, feat_w = self.image_token_grid
        obs_map = tokens.transpose(1, 2).reshape(bsz, channels, feat_h, feat_w)
        obs_map = self.obs_conv(obs_map)
        obs_map = F.adaptive_avg_pool2d(obs_map, (self.kernel_size, self.kernel_size))
        kernels = self.kernel_head(obs_map)
        return kernels


class OrienterNetLikelihoodModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.image_feature_dim = int(config.get("feature_dim", 128))
        self.map_feature_dim = int(config.get("orienternet_map_feature_dim", 64))
        self.theta_bins = int(config.get("orienternet_theta_bins", 36))
        self.kernel_size = int(config.get("orienternet_kernel_size", 9))
        if self.kernel_size % 2 != 1:
            raise ValueError("orienternet_kernel_size must be odd.")

        self.map_res = float(config["datasets"].get("map_res", 0.02))
        self.likelihood_loss = config.get("orienternet_likelihood_loss", "ce")
        self.target_label_smoothing = float(
            config.get("orienternet_label_smoothing", 0.0)
        )
        self.target_sigma_m = float(config.get("orienternet_target_sigma_m", 0.35))
        self.target_theta_sigma_deg = float(
            config.get("orienternet_target_theta_sigma_deg", 15.0)
        )
        self.gaussian_loss_weight = float(
            config.get("orienternet_gaussian_loss_weight", 0.25)
        )

        self.observation_encoder = ObservationKernelEncoder(
            image_encoder_name=config.get("image_encoder", "vits"),
            dptv2_ckpt_path=config.get(
                "dptv2_ckpt_path", "checkpoints/depth_anything_v2_vits.pth"
            ),
            image_feature_dim=self.image_feature_dim,
            map_feature_dim=self.map_feature_dim,
            theta_bins=self.theta_bins,
            kernel_size=self.kernel_size,
            image_token_grid=tuple(config.get("image_token_grid", [6, 40])),
            num_heads=int(config.get("num_heads", 4)),
            image_self_attn_layers=int(config.get("image_self_attn_layers", 1)),
            freeze_image_backbone=bool(config.get("freeze_image_backbone", True)),
        )
        map_encoder_type = config.get("orienternet_map_encoder", "cnn").lower()
        if map_encoder_type == "cnn":
            self.map_encoder = DenseFloorplanEncoder(
                feature_dim=self.map_feature_dim,
                output_stride=int(config.get("orienternet_map_output_stride", 8)),
                input_mode=config.get("orienternet_map_input_mode", "gray"),
                context_blocks=int(config.get("orienternet_map_context_blocks", 1)),
            )
        elif map_encoder_type == "resnet18":
            if int(config.get("orienternet_map_output_stride", 8)) != 8:
                raise ValueError("ResNetFloorplanEncoder currently requires output stride 8.")
            self.map_encoder = ResNetFloorplanEncoder(
                feature_dim=self.map_feature_dim,
                input_mode=config.get("orienternet_map_input_mode", "gray_edges"),
                context_blocks=int(config.get("orienternet_map_context_blocks", 1)),
                pretrained=bool(config.get("orienternet_map_pretrained", False)),
            )
        else:
            raise ValueError(
                f"Unsupported orienternet_map_encoder '{map_encoder_type}'. "
                "Expected one of: cnn, resnet18."
            )
        self.map_score_bias = nn.Conv2d(self.map_feature_dim, 1, kernel_size=1)
        nn.init.zeros_(self.map_score_bias.weight)
        nn.init.zeros_(self.map_score_bias.bias)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, obs_img: torch.Tensor, floorplan_img: torch.Tensor) -> torch.Tensor:
        map_feat = self.map_encoder(floorplan_img)
        kernels = self.observation_encoder(obs_img)
        bsz, channels, height, width = map_feat.shape

        kernels = kernels.reshape(
            bsz,
            self.theta_bins,
            self.map_feature_dim * self.kernel_size * self.kernel_size,
        )
        patches = F.unfold(
            map_feat,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        kernels = F.normalize(kernels, dim=-1)
        patches = F.normalize(patches, dim=1)
        logits = torch.einsum("btc,bcp->btp", kernels, patches)
        logits = logits.view(bsz, self.theta_bins, height, width)
        logits = logits + self.map_score_bias(map_feat)
        return logits * self.logit_scale.exp().clamp(max=100.0)

    def poses_to_targets(
        self,
        pose: torch.Tensor,
        wh: torch.Tensor,
        logit_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = logit_hw
        x = torch.floor(pose[:, 0] / wh[:, 0].clamp_min(1.0) * width).long()
        y = torch.floor(pose[:, 1] / wh[:, 1].clamp_min(1.0) * height).long()
        x = x.clamp(0, width - 1)
        y = y.clamp(0, height - 1)

        theta = torch.remainder(pose[:, 2], 2 * math.pi)
        theta_idx = torch.floor(theta / (2 * math.pi) * self.theta_bins).long()
        theta_idx = theta_idx.clamp(0, self.theta_bins - 1)
        flat = theta_idx * height * width + y * width + x
        return flat, x, y

    def build_gaussian_targets(
        self,
        pose: torch.Tensor,
        wh: torch.Tensor,
        logit_shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        batch_size, theta_bins, height, width = logit_shape
        device = pose.device
        dtype = pose.dtype

        center_x = pose[:, 0] / wh[:, 0].clamp_min(1.0) * width
        center_y = pose[:, 1] / wh[:, 1].clamp_min(1.0) * height

        # Convert meter sigma into each sample's likelihood-grid cell units.
        meters_per_cell_x = wh[:, 0].clamp_min(1.0) * self.map_res / width
        meters_per_cell_y = wh[:, 1].clamp_min(1.0) * self.map_res / height
        sigma_x = (self.target_sigma_m / meters_per_cell_x).clamp_min(1.0)
        sigma_y = (self.target_sigma_m / meters_per_cell_y).clamp_min(1.0)

        grid_x = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
        grid_y = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
        dx2 = (
            (grid_x + 0.5 - center_x.view(batch_size, 1, 1))
            / sigma_x.view(batch_size, 1, 1)
        ) ** 2
        dy2 = (
            (grid_y + 0.5 - center_y.view(batch_size, 1, 1))
            / sigma_y.view(batch_size, 1, 1)
        ) ** 2
        spatial = torch.exp(-0.5 * (dx2 + dy2))
        spatial = spatial / spatial.flatten(1).sum(dim=1).clamp_min(1e-8).view(
            batch_size, 1, 1
        )

        theta_centers = (
            torch.arange(theta_bins, device=device, dtype=dtype) + 0.5
        ) / theta_bins * (2 * math.pi)
        dtheta = torch.remainder(
            theta_centers.view(1, theta_bins) - pose[:, 2].view(batch_size, 1) + math.pi,
            2 * math.pi,
        ) - math.pi
        theta_sigma = math.radians(self.target_theta_sigma_deg)
        theta_target = torch.exp(-0.5 * (dtheta / max(theta_sigma, 1e-6)) ** 2)
        theta_target = theta_target / theta_target.sum(dim=1).clamp_min(1e-8).view(
            batch_size, 1
        )

        target = theta_target.view(batch_size, theta_bins, 1, 1) * spatial.unsqueeze(1)
        return target / target.flatten(1).sum(dim=1).clamp_min(1e-8).view(
            batch_size, 1, 1, 1
        )

    def likelihood_loss_value(
        self,
        logits: torch.Tensor,
        pose: torch.Tensor,
        wh: torch.Tensor,
    ) -> torch.Tensor:
        if self.likelihood_loss == "gaussian_kl":
            target = self.build_gaussian_targets(pose, wh, logits.shape)
            log_probs = F.log_softmax(logits.flatten(1), dim=1).view_as(logits)
            return -(target * log_probs).flatten(1).sum(dim=1).mean()

        target_flat, _target_x, _target_y = self.poses_to_targets(
            pose, wh, logits.shape[-2:]
        )
        ce_loss = F.cross_entropy(
            logits.flatten(1),
            target_flat,
            label_smoothing=self.target_label_smoothing,
        )
        if self.likelihood_loss == "ce_gaussian":
            target = self.build_gaussian_targets(pose, wh, logits.shape)
            log_probs = F.log_softmax(logits.flatten(1), dim=1).view_as(logits)
            gaussian_loss = -(target * log_probs).flatten(1).sum(dim=1).mean()
            return ce_loss + self.gaussian_loss_weight * gaussian_loss
        return ce_loss

    def predict_pose(
        self,
        logits: torch.Tensor,
        wh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, theta_bins, height, width = logits.shape
        pred = torch.argmax(logits.flatten(1), dim=1)
        theta_idx = pred // (height * width)
        rem = pred % (height * width)
        y = rem // width
        x = rem % width

        pred_x = (x.float() + 0.5) / width * wh[:, 0]
        pred_y = (y.float() + 0.5) / height * wh[:, 1]
        pred_theta = (theta_idx.float() + 0.5) / theta_bins * (2 * math.pi)
        return pred_x, pred_y, pred_theta

    @staticmethod
    def angular_error(pred_theta: torch.Tensor, gt_theta: torch.Tensor) -> torch.Tensor:
        return torch.abs(
            torch.remainder(pred_theta - gt_theta + math.pi, 2 * math.pi) - math.pi
        )

    def _shared_step(self, batch, batch_idx, stage: str):
        obs_img, pose, _ray, floorplan_img, wh, _local_map, _neg_local_map, _neg_pose = batch
        logits = self(obs_img, floorplan_img)
        loss = self.likelihood_loss_value(logits, pose, wh)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=obs_img.shape[0])

        with torch.no_grad():
            pred_x, pred_y, pred_theta = self.predict_pose(logits, wh)
            xy_err_m = (
                torch.sqrt((pred_x - pose[:, 0]) ** 2 + (pred_y - pose[:, 1]) ** 2)
                * self.map_res
            )
            theta_err = self.angular_error(pred_theta, pose[:, 2])
            recall_1m = (xy_err_m <= 1.0).float().mean()
            recall_05m = (xy_err_m <= 0.5).float().mean()
            recall_1m_30deg = (
                (xy_err_m <= 1.0) & (theta_err <= math.radians(30.0))
            ).float().mean()
            recall_1m_10deg = (
                (xy_err_m <= 1.0) & (theta_err <= math.radians(10.0))
            ).float().mean()

            self.log(
                f"{stage}_1m_recall",
                recall_1m,
                prog_bar=True,
                batch_size=obs_img.shape[0],
            )
            self.log(
                f"{stage}_0.5m_recall",
                recall_05m,
                prog_bar=False,
                batch_size=obs_img.shape[0],
            )
            self.log(
                f"{stage}_1m_30deg_recall",
                recall_1m_30deg,
                prog_bar=False,
                batch_size=obs_img.shape[0],
            )
            self.log(
                f"{stage}_1m_10deg_recall",
                recall_1m_10deg,
                prog_bar=False,
                batch_size=obs_img.shape[0],
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.config.get("lr", 1e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.get("epochs", 30)),
            eta_min=float(self.config.get("min_lr", 1e-6)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
