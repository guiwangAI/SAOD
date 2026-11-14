from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

from .data import box_semantic_target

class AcousticPromptEncoder(nn.Module):
    size = 256

    def __init__(self, nc: int):
        super().__init__()
        self.features = nn.Sequential(
            Conv(1, 16, 3, 2),
            Conv(16, 24, 3, 2),
            Conv(24, 48, 3, 2),
            DWConv(48, 48, 3, 2),
            Conv(48, 64, 1),
            DWConv(64, 64, 3),
            Conv(64, 64, 1),
        )
        self.semantic = nn.Conv2d(64, nc, 1)
        nn.init.constant_(self.semantic.bias, -2.0)

    def forward(self, image):
        if image.shape[-2:] != (self.size, self.size):
            image = F.interpolate(image, (self.size, self.size), mode="nearest-exact")
        features = self.features(image)
        return features, self.semantic(features)

class AcousticFeatureFusion(nn.Module):
    def __init__(self, channels: tuple[int, ...], nc: int, dimension: int = 16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError("Acoustic feature fusion requires PyTorch >= 2.0.")
        if len(channels) != 3:
            raise ValueError("Fusion requires three YOLO26 scales.")
        if not 0 < nc < dimension:
            raise ValueError("Fusion dimension must exceed the number of classes; UAOD has four classes.")
        self.dimension, self.nc = dimension, nc
        self.gate_kernel = 3
        self.query = nn.Conv2d(channels[1], dimension, 1)
        nn.init.zeros_(self.query.bias)
        nn.init.constant_(self.query.bias[-nc:], -2.0)
        self.key_value = nn.Conv2d(64, dimension * 2 - nc, 1, bias=False)
        self.local = nn.ModuleList(nn.Conv2d(1, 1, 3, padding=1, bias=False) for _ in channels)
        self.adapters = nn.ModuleList(nn.Conv2d(dimension, c + 2, 1) for c in channels)
        for channels_count, adapter in zip(channels, self.adapters):
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)
            nn.init.constant_(adapter.bias[channels_count + 1 :], -1.0)

    def forward(self, optical, sonar, semantic, present):
        query = self.query(optical[1])
        batch, _, height, width = query.shape
        content_dimension = self.dimension - self.nc
        query_content, query_semantic = query.split((content_dimension, self.nc), 1)
        key_content, values = self.key_value(sonar).split((content_dimension, self.dimension), 1)
        query_probability, key_probability = query_semantic.sigmoid(), semantic.sigmoid()
        probability = key_probability.amax(1).flatten(1)

        queries = torch.cat(
            (
                F.layer_norm(query_content.flatten(2).transpose(1, 2), (content_dimension,)),
                query_probability.flatten(2).transpose(1, 2) * 2.0,
            ),
            -1,
        ).contiguous()
        keys = torch.cat(
            (
                F.layer_norm(key_content.flatten(2).transpose(1, 2), (content_dimension,)),
                key_probability.flatten(2).transpose(1, 2) * 2.0,
            ),
            -1,
        ).contiguous()
        values = values.flatten(2).transpose(1, 2).contiguous()
        message = (
            F.scaled_dot_product_attention(
                queries[:, None], keys[:, None], (values * probability[..., None])[:, None], dropout_p=0.0
            )[:, 0]
            .transpose(1, 2)
            .reshape(batch, self.dimension, height, width)
        )
        residuals = []
        for feature, local, adapter in zip(optical, self.local, self.adapters):
            context = message
            if context.shape[-2:] != feature.shape[-2:]:
                context = F.interpolate(context, feature.shape[-2:], mode="bilinear", align_corners=False)
            shift, scale, gate = adapter(context).split((feature.shape[1], 1, 1), 1)
            gate = gate + local(feature.mean(1, keepdim=True))
            residual = feature * scale.tanh() + shift
            residuals.append(residual * gate.sigmoid() * present[:, None, None, None])
        return residuals, query_semantic

class AcousticAssistDetect(Detect):
    def __init__(self, original: Detect, dimension=16):
        channels = tuple(block[0].conv.in_channels for block in original.cv2)
        super().__init__(original.nc, original.reg_max, hasattr(original, "one2one_cv2"), channels)
        self.load_state_dict(original.state_dict())
        for attribute in ("i", "f", "type", "stride", "inplace", "legacy"):
            if hasattr(original, attribute):
                setattr(self, attribute, deepcopy(getattr(original, attribute)))
        self.end2end = original.end2end
        self.encoder = AcousticPromptEncoder(original.nc)
        self.fusion = AcousticFeatureFusion(channels, original.nc, dimension)
        self.mode = "full"
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features, acoustic=None):
        semantic = query_semantic = present = None
        if acoustic is not None:
            sonar, semantic = self.encoder(acoustic)
            present = (acoustic.detach().abs().amax((1, 2, 3)) > 0).to(sonar.dtype)

            residuals, query_semantic = self.fusion(
                [feature.detach() for feature in features], sonar, semantic, present
            )
            enhanced = [feature + residual for feature, residual in zip(features, residuals)]
        else:
            enhanced = features
        if self.training and self.end2end and semantic is not None:
            predictions = {
                "one2many": self.forward_head(enhanced, **self.one2many),
                "one2one": self.forward_head(
                    [feature.detach() + residual for feature, residual in zip(features, residuals)], **self.one2one
                ),
            }
        else:
            predictions = super().forward(enhanced)
        if semantic is not None and not self.export:
            raw = predictions if self.training else predictions[1]
            raw["acoustic_semantic_logits"] = semantic
            raw["optical_prompt_logits"] = query_semantic
            raw["acoustic_present"] = present
        return predictions

class AcousticAssistedModel(DetectionModel):
    def __init__(
        self, cfg="yolo26n.yaml", nc=None, verbose=True, dimension=16,
        acoustic_loss_weight=0.1, optical_prompt_loss_weight=0.05,
    ):
        super().__init__(deepcopy(cfg), ch=3, nc=nc, verbose=False)
        if type(self.model[-1]) is not Detect:
            raise ValueError("Use a standard YOLO26 detection configuration.")
        self.model[-1] = AcousticAssistDetect(self.model[-1], dimension)
        self.acoustic_loss_weight = float(acoustic_loss_weight)
        self.optical_prompt_loss_weight = float(optical_prompt_loss_weight)
        if any(not math.isfinite(w) or w < 0 for w in (self.acoustic_loss_weight, self.optical_prompt_loss_weight)):
            raise ValueError("Semantic loss weights must be finite and nonnegative.")
        self.yaml["channels"] = 4
        self.yaml["sonar_assist"] = {
            "dimension": dimension, "mode": "full", "version": 4, "gate_kernel": 3,
            "acoustic_loss_weight": self.acoustic_loss_weight,
            "optical_prompt_loss_weight": self.optical_prompt_loss_weight,
        }
        if verbose:
            self.info()

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None, acoustic=None):
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed, acoustic)

    def _predict_once(self, x, profile=False, visualize=False, embed=None, acoustic=None):
        if profile or visualize or embed:
            raise ValueError("Layer profiling and embedding are unsupported for paired inputs.")
        if x.ndim != 4 or x.shape[1] not in (3, 4):
            raise ValueError("Expected BCHW RGB, optionally with an independent acoustic carrier channel.")
        acoustic = x[:, 3:4] if x.shape[1] == 4 else acoustic
        x, saved = x[:, :3], []
        for module in self.model:
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f, int) else [x if j == -1 else saved[j] for j in module.f]
            x = module(x, acoustic) if isinstance(module, AcousticAssistDetect) else module(x)
            saved.append(x if module.i in self.save else None)
        return x

    def _predict_augment(self, x):
        raise ValueError("Paired inference uses one optical scale, without spatial TTA.")

    def info(self, detailed=False, verbose=True, imgsz=640):
        layers = len(list(self.modules()))
        parameters = sum(value.numel() for value in self.parameters())
        gradients = sum(value.numel() for value in self.parameters() if value.requires_grad)
        if verbose:
            LOGGER.info(
                f"SAOD YOLO26: {layers} layers, {parameters:,} parameters."
            )
        return layers, parameters, gradients, 0.0

    def loss(self, batch, preds=None):
        preds = self.forward(batch["img"]) if preds is None else preds
        loss, items = super().loss(batch, preds)
        raw = preds[1] if isinstance(preds, tuple) else preds
        acoustic_loss = prompt_loss = loss.new_zeros(())
        if "acoustic_semantic_logits" in raw:
            logits = raw["acoustic_semantic_logits"].float()
            target = batch["acoustic_semantic"].to(device=logits.device, dtype=logits.dtype)
            valid = batch["acoustic_valid"].to(device=logits.device, dtype=logits.dtype)
            valid = valid * raw["acoustic_present"][:, None, None, None]
            values = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=logits.new_tensor(8.0), reduction="none"
            )
            acoustic_loss = (values * valid).sum() / (valid.sum() * logits.shape[1]).clamp_min(1)
            acoustic_loss = acoustic_loss * self.acoustic_loss_weight
            query_logits = raw["optical_prompt_logits"].float()
            query_target = box_semantic_target(batch["bboxes"], batch["cls"], batch["batch_idx"], query_logits)
            query_valid = raw["acoustic_present"][:, None, None, None].expand_as(query_logits)
            values = F.binary_cross_entropy_with_logits(
                query_logits, query_target, pos_weight=query_logits.new_tensor(8.0), reduction="none"
            )
            prompt_loss = (values * query_valid).sum() / query_valid.sum().clamp_min(1)
            prompt_loss = prompt_loss * self.optical_prompt_loss_weight
        auxiliary = torch.stack((acoustic_loss, prompt_loss))
        return torch.cat((loss, auxiliary * batch["img"].shape[0])), torch.cat((items, auxiliary.detach()))
