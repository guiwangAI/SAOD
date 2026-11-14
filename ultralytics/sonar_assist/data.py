from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ultralytics.data.augment import LetterBox, Mosaic
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.tal import make_anchors

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

@dataclass(frozen=True)
class ImagePair:
    key: str
    optical: Path
    acoustic: Path | None
    optical_label: Path
    acoustic_label: Path | None

def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode image: {path}. Check empty files and broken symbolic links.")
    return image

def read_labels(path: Path, nc: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = np.asarray([float(x) for x in line.split()], dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid detection annotation at {path}:{line_number}")
        category, x, y, w, h = values
        if category != int(category) or not 0 <= category < nc:
            raise ValueError(f"Invalid category at {path}:{line_number}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized box at {path}:{line_number}")
        rows.append(values)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)

def canonical_names(names) -> list[str]:
    return [" ".join(str(name).lower().replace("_", " ").split()) for name in names]

class PairedDataset:
    def __init__(self, config: str | Path, root: str | Path | None = None):
        self.config_path = Path(config).resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        configured_root = Path(root) if root is not None else Path(self.config["root"])
        self.root = (
            configured_root.resolve() if root is not None else (self.config_path.parent / configured_root).resolve()
        )
        names = self.config["names"]
        if isinstance(names, dict) and sorted(map(int, names)) != list(range(len(names))):
            raise ValueError("Category IDs must be contiguous and start at zero.")
        self.names = list(names) if isinstance(names, list) else [names[k] for k in sorted(names, key=int)]
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Category names must be nonempty and unique.")
        self.nc = len(self.names)

    def image_dir(self, modality: str, split: str) -> Path:
        if split not in {"train", "val", "test"}:
            raise ValueError("Only released train/val/test splits are supported.")
        return self.root / self.config["modalities"][modality]["images"] / split

    def label_dir(self, modality: str, split: str) -> Path:
        return self.root / self.config["modalities"][modality]["labels"] / split

    @staticmethod
    def image_index(directory: Path) -> dict[str, Path]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
        index = {}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            if path.stem in index:
                raise ValueError(f"Ambiguous image stem {path.stem!r} in {directory}")
            if not path.stat().st_size:
                raise ValueError(f"Zero-byte image: {path}. The original archive contains some empty placeholders.")
            index[path.stem] = path
        return index

    def pairs(self, split: str, allow_missing_acoustic: bool = False, require_labels: bool = False) -> list[ImagePair]:
        optical = self.image_index(self.image_dir("optical", split))
        acoustic_dir = self.image_dir("acoustic", split)
        acoustic = {} if allow_missing_acoustic and not acoustic_dir.is_dir() else self.image_index(acoustic_dir)
        if not optical:
            raise ValueError(f"No optical images in {split}.")
        missing = optical.keys() - acoustic.keys()
        extra = acoustic.keys() - optical.keys()
        if extra or (missing and not allow_missing_acoustic):
            raise ValueError(
                f"{split}: {len(missing)} missing acoustic images, {len(extra)} unmatched acoustic images."
            )
        pairs = [
            ImagePair(
                f"{split}/{stem}",
                image,
                acoustic.get(stem),
                self.label_dir("optical", split) / f"{stem}.txt",
                self.label_dir("acoustic", split) / f"{stem}.txt" if stem in acoustic else None,
            )
            for stem, image in optical.items()
        ]
        if require_labels:
            for pair in pairs:
                read_labels(pair.optical_label, self.nc)
                if pair.acoustic_label is not None:
                    read_labels(pair.acoustic_label, self.nc)
        return pairs

    def yolo_yaml(self, destination: Path, options: dict | None = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "path": str(self.root),
            "train": str(self.image_dir("optical", "train")),
            "val": str(self.image_dir("optical", "val")),
            "test": str(self.image_dir("optical", "test")),
            "names": dict(enumerate(self.names)),
            "channels": 4,
        }
        content.update(paired_config=str(self.config_path), paired_root=str(self.root), assist_options=options or {})
        destination.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return destination

def acoustic_canvas(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return LetterBox((256, 256), padding_value=0)(image=gray).reshape(256, 256)

@torch.no_grad()
def box_semantic_target(boxes, categories, batch_index, logits) -> torch.Tensor:
    batch, nc, height, width = logits.shape
    boxes = boxes.to(device=logits.device, dtype=torch.float32)
    categories = categories.to(device=logits.device, dtype=torch.long).flatten()
    batch_index = batch_index.to(device=logits.device, dtype=torch.long).flatten()
    target = logits.float().new_zeros(batch, nc, height * width)
    centers, _ = make_anchors([logits], [1], 0.5)
    centers = centers.float() / centers.new_tensor((width, height))
    distance = (centers[None] - boxes[:, None, :2]).abs()
    covered = (distance <= boxes[:, None, 2:] / 2).all(-1)
    rows = torch.arange(len(boxes), device=logits.device)
    closest = distance.square().sum(-1).argmin(-1)
    covered[rows, closest] |= ~covered.any(-1)
    rows, cells = covered.nonzero(as_tuple=True)
    target[batch_index[rows], categories[rows], cells] = 1
    return target.reshape(batch, nc, height, width)

def foreground_target(labels: np.ndarray, shape: tuple, nc: int) -> torch.Tensor:
    h, w = shape[:2]
    ratio = min(256 / h, 256 / w)
    resized_w, resized_h = round(w * ratio), round(h * ratio)
    pad = torch.tensor([round((256 - resized_w) / 2 - 0.1), round((256 - resized_h) / 2 - 0.1)])
    boxes = torch.as_tensor(labels[:, 1:5], dtype=torch.float32).clone()
    boxes[:, :2] = (boxes[:, :2] * torch.tensor((w, h)) * ratio + pad) / 256
    boxes[:, 2:] *= torch.tensor((w, h)) * ratio / 256
    return box_semantic_target(
        boxes, torch.as_tensor(labels[:, 0]), torch.zeros(len(labels)), torch.zeros(1, nc, 16, 16)
    )[0]

class PairedMosaic(Mosaic):
    def _cat_labels(self, mosaic_labels):
        labels = super()._cat_labels(mosaic_labels)
        if len(mosaic_labels) != 4:
            raise ValueError("Acoustic transport supports the native four-source Mosaic.")
        image = np.zeros((256, 256), dtype=np.uint8)
        target = torch.zeros_like(mosaic_labels[0]["acoustic_semantic"])
        valid = torch.zeros((1, 16, 16))
        for index, source in enumerate(mosaic_labels):
            row, column = divmod(index, 2)
            image[row * 128 : (row + 1) * 128, column * 128 : (column + 1) * 128] = cv2.resize(
                source["acoustic_image"], (128, 128), interpolation=cv2.INTER_AREA
            )
            target[:, row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8] = torch.nn.functional.max_pool2d(
                source["acoustic_semantic"], 2
            )
            valid[:, row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8] = torch.nn.functional.max_pool2d(
                source["acoustic_valid"], 2
            )
        labels.update(acoustic_image=image, acoustic_semantic=target, acoustic_valid=valid)
        return labels

class PairedYOLODataset(YOLODataset):
    def __init__(self, *args, data: dict, **kwargs):
        self.paired = PairedDataset(data["paired_config"], data["paired_root"])
        split = Path(kwargs["img_path"]).name
        if split not in {"train", "val", "test"}:
            raise ValueError("Joint directories must retain train/val/test names.")
        pairs = self.paired.pairs(split, allow_missing_acoustic=True, require_labels=False)
        self.acoustics = {str(pair.optical.resolve()): pair for pair in pairs}
        self.dropout = float(data.get("assist_options", {}).get("dropout", 0.05))
        if not 0 <= self.dropout <= 1:
            raise ValueError("Acoustic dropout must be in [0, 1].")
        super().__init__(*args, data=dict(data, channels=3), **kwargs)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            pre_transform = transforms.transforms[0]
            for index, transform in enumerate(pre_transform.transforms):
                if isinstance(transform, Mosaic):
                    pre_transform.transforms[index] = PairedMosaic(self, self.imgsz, p=transform.p, n=transform.n)
        return transforms

    def get_image_and_label(self, index: int) -> dict:
        sample = super().get_image_and_label(index)
        pair = self.acoustics[str(Path(sample["im_file"]).resolve())]
        canvas = np.zeros((256, 256), dtype=np.uint8)
        target = torch.zeros((self.paired.nc, 16, 16))
        valid = torch.zeros((1, 16, 16))
        if pair.acoustic is not None:
            image = read_image(pair.acoustic)
            canvas = acoustic_canvas(image)
            if pair.acoustic_label.is_file():
                target = foreground_target(
                    read_labels(pair.acoustic_label, self.paired.nc), image.shape, self.paired.nc
                )
                valid.fill_(1)
            elif self.augment:
                raise FileNotFoundError(f"Missing acoustic training annotation: {pair.acoustic_label}")
        if self.augment and random.random() < self.dropout:
            canvas.fill(0)
            valid.zero_()
        sample.update(acoustic_image=canvas, acoustic_semantic=target, acoustic_valid=valid)
        return sample

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        height, width = sample["img"].shape[-2:]
        carrier = cv2.resize(sample.pop("acoustic_image"), (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
        sample["img"] = torch.cat((sample["img"], torch.from_numpy(carrier[None])), 0)
        return sample

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        result = YOLODataset.collate_fn(batch)
        for key in ("acoustic_semantic", "acoustic_valid"):
            result[key] = torch.stack(result[key])
        return result
