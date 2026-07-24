import torch.nn as nn
import torch.nn.functional as F

from RRP_model.RRP import RRPFeatureExtractor
from RRP_model.models import F3MlpDecoder


class DepthPredModels(nn.Module):
    def __init__(self, config, encoder_type="dptv2", decoder_type="f3mlp"):
        super().__init__()
        if encoder_type != "dptv2":
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        if decoder_type not in ("f3mlp", "rrp"):
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

        self.config = config
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self._init_encoders()
        self._init_decoders()

    def forward(self, func_name, **kwargs):
        if func_name == "encode":
            return self._encode(**kwargs)
        if func_name == "decoder_train":
            return self._decoder_train_get_pred_loss(
                cond=kwargs["depth_cond"],
                gt_ray=kwargs["gt_ray"],
            )
        if func_name == "decoder_inference":
            return self._decoder_inference_get_pred(cond=kwargs["depth_cond"])
        raise ValueError(f"Unsupported model operation: {func_name}")

    def _encode(self, obs_img):
        features, _, _ = self.dptv2_encoder(obs_img=obs_img)
        return features

    def _decoder_train_get_pred_loss(self, cond, gt_ray):
        prediction = self._decoder()(cond)
        return {
            "pred": prediction,
            "loss": F.l1_loss(prediction, gt_ray),
        }

    def _decoder_inference_get_pred(self, cond):
        return self._decoder()(cond)

    def _init_encoders(self):
        self.dptv2_encoder = RRPFeatureExtractor(
            checkpoint_path=self.config["dptv2_ckpt_path"]
        )

    def _init_decoders(self):
        if self.decoder_type == "f3mlp":
            self.f3mlp_decoder = F3MlpDecoder()
        else:
            self.rrp_decoder = F3MlpDecoder()

    def _decoder(self):
        if self.decoder_type == "f3mlp":
            return self.f3mlp_decoder
        return self.rrp_decoder
