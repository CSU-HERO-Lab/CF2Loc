import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tv_functional

from DisCo_model.depth_anything_v2.dinov2 import DINOv2
from DisCo_model.network_utils import ConvBnReLU


class RRPFeatureExtractor(nn.Module):
    def __init__(
        self,
        encoder="vits",
        embed_dim=128,
        pos_embed_dim=32,
        num_heads=8,
        target_size=(23, 40),
        checkpoint_path=None,
    ):
        super().__init__()
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for RRPFeatureExtractor.")

        self.embed_dim = embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.intermediate_layer_idx = 11
        self.num_heads = num_heads

        params = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.pretrained = DINOv2(model_name=encoder)
        pretrained_dict = OrderedDict(
            (key[len("pretrained.") :], value)
            for key, value in params.items()
            if key.startswith("pretrained.")
        )
        self.pretrained.load_state_dict(pretrained_dict, strict=True)
        self.pretrained.requires_grad_(False)

        self.conv = ConvBnReLU(
            in_channels=384,
            out_channels=self.embed_dim,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.vertical_pool = VerticalAttentionPooling(in_channels=self.embed_dim)

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_embed_dim, self.pos_embed_dim),
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_embed_dim, self.pos_embed_dim),
        )

        total_embed_dim = self.embed_dim + self.pos_embed_dim
        self.q_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.target_size = target_size

    def forward(self, obs_img, mask=None):
        batch_size, _channels, height, width = obs_img.shape
        patch_size = 14
        target_h = int(math.ceil(height / patch_size) * patch_size)
        target_w = int(math.ceil(width / patch_size) * patch_size)
        padded = F.pad(
            obs_img,
            (0, target_w - width, 0, target_h - height),
        )

        features = self.pretrained.get_intermediate_layers(
            padded,
            [self.intermediate_layer_idx],
            return_class_token=True,
        )
        patch_tokens, class_token = features[0]
        grid_h = target_h // patch_size
        grid_w = target_w // patch_size
        features_2d = patch_tokens.permute(0, 2, 1).reshape(
            batch_size,
            384,
            grid_h,
            grid_w,
        )
        features_2d = F.interpolate(
            features_2d,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        features_2d = self.conv(features_2d)

        pooled, _pooling_weights = self.vertical_pool(features_2d)
        query = pooled.permute(0, 2, 1)
        spatial_tokens = features_2d.flatten(2).permute(0, 2, 1)

        pos_x = (
            torch.linspace(
                0,
                1,
                self.target_size[1],
                device=spatial_tokens.device,
            )
            - 0.5
        )
        pos_y = (
            torch.linspace(
                0,
                1,
                self.target_size[0],
                device=spatial_tokens.device,
            )
            - 0.5
        )
        pos_y_grid, pos_x_grid = torch.meshgrid(
            pos_y,
            pos_x,
            indexing="ij",
        )
        pos_2d = torch.stack((pos_x_grid, pos_y_grid), dim=-1)
        pos_2d = self.pos_mlp_2d(pos_2d).reshape(
            1,
            -1,
            self.pos_embed_dim,
        )
        pos_2d = pos_2d.expand(batch_size, -1, -1)
        spatial_tokens = torch.cat((spatial_tokens, pos_2d), dim=-1)

        pos_1d = (
            torch.linspace(
                0,
                1,
                self.target_size[1],
                device=query.device,
            )
            - 0.5
        )
        pos_1d = self.pos_mlp_1d(pos_1d.unsqueeze(-1)).unsqueeze(0)
        pos_1d = pos_1d.expand(batch_size, -1, -1)
        query = torch.cat((query, pos_1d), dim=-1)

        query = self.q_proj(query)
        key = self.k_proj(spatial_tokens)
        value = self.v_proj(spatial_tokens)

        key_padding_mask = None
        if mask is not None:
            resized_mask = tv_functional.resize(
                mask,
                self.target_size,
                tv_functional.InterpolationMode.NEAREST,
            ).bool()
            key_padding_mask = ~resized_mask.reshape(batch_size, -1)

        output, attention_weights = self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
        )
        return output, attention_weights, class_token

    def train(self, mode=True):
        super().train(mode)
        self.pretrained.eval()
        return self


class VerticalAttentionPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x, mask=None):
        del mask
        scores = self.net(x)
        attention_weights = F.softmax(scores, dim=2)
        pooled = (x * attention_weights).sum(dim=2)
        return pooled, attention_weights
