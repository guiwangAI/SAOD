from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import torch

from ultralytics.utils.ops import xyxy2xywh

from .data import PairedDataset, canonical_names, read_image
from .pipeline import JointPredictor
from .trainer import PairedDetectionTrainer

def save_prediction(path, detections, image, names, draw=False):
    height, width = image.shape[:2]
    boxes = xyxy2xywh(detections[:, :4].copy()) / [width, height, width, height]
    label_path = path.parent / "labels" / f"{path.stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(
            f"{int(row[5])} " + " ".join(f"{value:.6f}" for value in (*box, row[4]))
            for row, box in zip(detections, boxes)
        ),
        encoding="utf-8",
    )
    if draw:
        canvas = image.copy()
        for row in detections:
            x1, y1, x2, y2 = row[:4].astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(
                canvas,
                f"{names[int(row[5])]} {row[4]:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 255),
                1,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), canvas):
            raise OSError(f"Cannot write prediction image: {path}")

def train(args, dataset):
    for split in ("train", "val"):
        dataset.pairs(split, require_labels=True)
    project = Path(args.project)
    options = {
        "dimension": 16,
        "dropout": args.sonar_dropout,
        "acoustic_loss_weight": args.acoustic_loss_weight,
        "optical_prompt_loss_weight": args.optical_prompt_loss_weight,
    }
    yaml_path = dataset.yolo_yaml(project / "paired_data.yaml", options)
    overrides = {
        "model": args.weights,
        "data": str(yaml_path.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "imgsz": args.imgsz,
        "device": args.device,
        "seed": args.seed,
        "deterministic": True,
        "project": str(project),
        "name": "joint",
        "optimizer": "auto",
        "cache": False,
        "plots": False,
        "end2end": args.end2end,
        "amp": args.amp,
        "pretrained": not args.scratch,
        "resume": args.resume or False,
    }
    PairedDetectionTrainer(overrides=overrides).train()

def predict(args, dataset):
    project = Path(args.project)
    checkpoint = args.checkpoint or project / "joint" / "weights" / "best.pt"
    predictor = JointPredictor(checkpoint, args.imgsz, args.device, args.conf, args.half)
    if canonical_names(predictor.names) != canonical_names(dataset.names):
        raise ValueError("Checkpoint categories do not match the data configuration.")
    destination = Path(args.output or project / f"predictions_{args.split}")
    pairs = dataset.pairs(args.split, allow_missing_acoustic=True)
    for pair in pairs[: args.limit or len(pairs)]:
        optical = read_image(pair.optical)
        acoustic = read_image(pair.acoustic) if pair.acoustic is not None else None
        detections = predictor(optical, acoustic)
        save_prediction(destination / f"{pair.optical.stem}.png", detections, optical, dataset.names, args.save_images)

def main():
    parser = argparse.ArgumentParser(description="SAOD acoustic-optical detection")
    parser.add_argument("command", choices=("train", "predict"))
    parser.add_argument("--data", default="configs/uaod.yaml")
    parser.add_argument("--root")
    parser.add_argument("--project", default="runs/saod")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--acoustic-loss-weight", type=float, default=0.1)
    parser.add_argument("--optical-prompt-loss-weight", type=float, default=0.05)
    parser.add_argument("--sonar-dropout", type=float, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0 for value in (args.acoustic_loss_weight, args.optical_prompt_loss_weight)):
        parser.error("Semantic loss weights must be finite and nonnegative.")
    if not 0 <= args.sonar_dropout <= 1:
        parser.error("Sonar dropout must be in [0, 1].")
    if args.limit < 0 or args.imgsz < 32 or args.imgsz % 32 or args.threads < 1:
        parser.error("Invalid image size, thread count, or image limit.")
    if not 0 <= args.conf <= 1:
        parser.error("Confidence threshold must be in [0, 1].")
    if args.resume and args.command != "train":
        parser.error("Resume is supported only for training.")
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    dataset = PairedDataset(args.data, args.root)
    if args.command == "train":
        train(args, dataset)
    else:
        predict(args, dataset)

if __name__ == "__main__":
    main()
