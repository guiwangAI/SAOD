from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model

from .data import PairedYOLODataset
from .model import AcousticAssistedModel

def build_paired_dataset(args, image_path, batch, data, mode, stride=32):
    if mode == "train" and any(getattr(args, field, 0) != 0 for field in ("mixup", "cutmix", "copy_paste")):
        raise ValueError("Paired training supports native Mosaic/geometry; other mixing is unsupported.")
    return PairedYOLODataset(
        img_path=image_path,
        imgsz=args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=args,
        rect=mode == "val",
        cache=False,
        single_cls=args.single_cls,
        stride=stride,
        pad=0 if mode == "train" else 0.5,
        task="detect",
        classes=args.classes,
        data=data,
        fraction=args.fraction if mode == "train" else 1.0,
    )

class PairedDetectionValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        return build_paired_dataset(self.args, img_path, batch, self.data, mode, int(self.stride))

class PairedDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        options = self.data.get("assist_options", {})
        stored = cfg.get("sonar_assist", {}) if isinstance(cfg, dict) else {}
        model = self.set_model_names_for_load(
            AcousticAssistedModel(
                cfg,
                nc=self.data["nc"],
                verbose=verbose,
                dimension=options.get("dimension", stored.get("dimension", 16)),
                acoustic_loss_weight=options.get("acoustic_loss_weight", stored.get("acoustic_loss_weight", 0.1)),
                optical_prompt_loss_weight=options.get(
                    "optical_prompt_loss_weight", stored.get("optical_prompt_loss_weight", 0.05)
                ),
            )
        )
        if weights is not None:
            model.load(weights)
        if self.args.end2end is not None:
            model.end2end = self.args.end2end
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        return build_paired_dataset(
            self.args, img_path, batch, self.data, mode, int(unwrap_model(self.model).stride.max())
        )

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "acoustic_semantic_loss", "optical_prompt_loss")
        return PairedDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
