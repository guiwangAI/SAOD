from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from .data import acoustic_canvas
from .model import AcousticAssistedModel

class JointPredictor:
    def __init__(self, weights: str | Path, imgsz=640, device="0", conf=0.25, half=False):
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.weights}")
        self.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.model, _ = load_checkpoint(str(self.weights), device=self.device, fuse=False)
        if not isinstance(self.model, AcousticAssistedModel):
            raise TypeError("Use a SAOD detection checkpoint.")
        if self.model.yaml.get("sonar_assist", {}).get("version") != 4:
            raise ValueError("v2/v3 checkpoints need their original code. Retrain v4 in a new output directory.")
        self.end2end = bool(self.model.end2end)
        self.model = self.model.float().eval().fuse(verbose=False)
        if half:
            if self.device.type != "cuda":
                raise ValueError("FP16 prediction requires CUDA; use FP32 on CPU.")
            self.model = self.model.half()
        self.dtype = next(self.model.parameters()).dtype
        self.imgsz, self.conf = imgsz, conf
        self.names = [self.model.names[i] for i in sorted(self.model.names)]

    def prepare(self, optical: np.ndarray, acoustic: np.ndarray | None):
        canvas = LetterBox((self.imgsz, self.imgsz))(image=optical)
        rgb = np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(rgb)[None].to(self.device, dtype=self.dtype) / 255
        sonar = None
        if acoustic is not None:
            sonar = torch.from_numpy(acoustic_canvas(acoustic))[None, None].to(self.device, dtype=self.dtype) / 255
        return tensor, sonar

    @torch.inference_mode()
    def __call__(self, optical: np.ndarray, acoustic: np.ndarray | None = None) -> np.ndarray:
        tensor, sonar = self.prepare(optical, acoustic)
        prediction = self.model(tensor, acoustic=sonar)
        detections = non_max_suppression(
            prediction, conf_thres=self.conf, iou_thres=0.7, nc=len(self.names), end2end=self.end2end
        )[0]
        detections[:, :4] = scale_boxes(tensor.shape[-2:], detections[:, :4], optical.shape)
        return detections.float().cpu().numpy().astype(np.float64)
