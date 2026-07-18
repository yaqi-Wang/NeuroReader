import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data_multitarget import infer_token_spec, prepare_fmri_batch
from models_multitarget import TokenSpec, build_stage1_model, token_mse_cosine_loss

import utils


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class Stage1ClipTrainConfig:
    train_fmri_path: str
    train_janus_vision_path: str
    train_clip_vision_path: str
    train_clip_text_path: str
    eval_fmri_path: str
    eval_janus_vision_path: str
    eval_clip_vision_path: str
    eval_clip_text_path: str
    output_dir: str
    num_voxels: int
    train_ids_path: Optional[str] = None
    eval_ids_path: Optional[str] = None
    batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 0
    num_epochs: int = 80
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    lambda_janus: float = 1.0
    lambda_clip_vision: float = 0.3
    lambda_clip_text: float = 0.1
    hidden_dim: int = 4096
    num_blocks: int = 4
    shared_token_count: Optional[int] = None
    shared_token_dim: Optional[int] = None
    head_hidden_dim: int = 2048
    dropout: float = 0.1
    seed: int = 42
    mixed_precision: str = "bf16"
    save_every: int = 5
    resume_from: Optional[str] = None
    subject: Optional[str] = None
    roi: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Data — CLIP variant dataset
# ──────────────────────────────────────────────────────────────

@dataclass
class ClipMultiTargetDataConfig:
    train_fmri_path: str
    train_janus_vision_path: str
    train_clip_vision_path: str
    train_clip_text_path: str
    eval_fmri_path: str
    eval_janus_vision_path: str
    eval_clip_vision_path: str
    eval_clip_text_path: str
    train_ids_path: Optional[str] = None
    eval_ids_path: Optional[str] = None
    batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 0
    shuffle_train: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = 2
    seed: int = 42


class ClipMultiTargetDataset(Dataset):
    def __init__(
        self,
        fmri_path: str,
        janus_vision_path: str,
        clip_vision_path: str,
        clip_text_path: str,
        ids_path: Optional[str] = None,
    ) -> None:
        self.fmri = np.load(fmri_path, mmap_mode="r")
        self.janus_vision = np.load(janus_vision_path, mmap_mode="r")
        self.clip_vision = np.load(clip_vision_path, mmap_mode="r")
        self.clip_text = np.load(clip_text_path, mmap_mode="r")
        self.sample_ids = np.load(ids_path, mmap_mode="r") if ids_path else None

        total = int(self.fmri.shape[0])
        for name, array in [("janus_vision", self.janus_vision), ("clip_vision", self.clip_vision), ("clip_text", self.clip_text)]:
            if int(array.shape[0]) != total:
                raise ValueError(f"fMRI and {name} sample counts do not match: {total} vs {array.shape[0]}")
        if self.sample_ids is not None and int(self.sample_ids.shape[0]) != total:
            raise ValueError("fMRI and sample id counts do not match")

    def __len__(self) -> int:
        return int(self.fmri.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = {
            "fmri": torch.from_numpy(np.asarray(self.fmri[index], dtype=np.float32).copy()),
            "janus_vision": torch.from_numpy(np.asarray(self.janus_vision[index], dtype=np.float32).copy()),
            "clip_vision": torch.from_numpy(np.asarray(self.clip_vision[index], dtype=np.float32).copy()),
            "clip_text": torch.from_numpy(np.asarray(self.clip_text[index], dtype=np.float32).copy()),
        }
        if self.sample_ids is not None:
            sample["sample_id"] = torch.from_numpy(np.asarray(self.sample_ids[index]).copy())
        else:
            sample["sample_id"] = torch.tensor(index, dtype=torch.long)
        return sample


def _infer_num_voxels(fmri_path: str) -> int:
    fmri = np.load(fmri_path, mmap_mode="r")
    if fmri.ndim < 2:
        raise ValueError(f"Expected fMRI array with at least 2 dims, got shape {fmri.shape}")
    return int(fmri.shape[-1])


def _subject_to_num(subject: str) -> int:
    digits = "".join(ch for ch in subject if ch.isdigit())
    if not digits:
        raise ValueError(f"Unable to parse numeric subject id from '{subject}'")
    return int(digits)


def _format_value(value, subject: str, roi: str):
    if not isinstance(value, str):
        return value
    subject_num = _subject_to_num(subject)
    return value.format(
        subject=subject,
        subject_num=subject_num,
        subject_num_2d=f"{subject_num:02d}",
        sub=f"sub{subject_num}",
        roi=roi,
    )


def _load_config_file(path: Optional[str]) -> Dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            return json.load(handle)
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML config files") from exc
        return yaml.safe_load(handle) or {}


def _build_single_config(args: argparse.Namespace, file_config: Dict, subject: Optional[str] = None, roi: Optional[str] = None) -> Stage1ClipTrainConfig:
    merged = {
        "train_fmri_path": file_config.get("train_fmri_path", args.train_fmri_path),
        "train_janus_vision_path": file_config.get("train_janus_vision_path", args.train_janus_vision_path),
        "train_clip_vision_path": file_config.get("train_clip_vision_path", args.train_clip_vision_path),
        "train_clip_text_path": file_config.get("train_clip_text_path", args.train_clip_text_path),
        "eval_fmri_path": file_config.get("eval_fmri_path", file_config.get("val_fmri_path", args.eval_fmri_path)),
        "eval_janus_vision_path": file_config.get("eval_janus_vision_path", file_config.get("val_janus_vision_path", args.eval_janus_vision_path)),
        "eval_clip_vision_path": file_config.get("eval_clip_vision_path", file_config.get("val_clip_vision_path", args.eval_clip_vision_path)),
        "eval_clip_text_path": file_config.get("eval_clip_text_path", file_config.get("val_clip_text_path", args.eval_clip_text_path)),
        "num_voxels": file_config.get("num_voxels", args.num_voxels),
        "output_dir": file_config.get("output_dir", args.output_dir),
    }

    if subject is not None and roi is not None:
        merged = {key: _format_value(value, subject, roi) for key, value in merged.items()}
        if file_config.get("num_voxels") is None and args.num_voxels is None:
            merged["num_voxels"] = _infer_num_voxels(merged["train_fmri_path"])

    missing = [key for key, value in merged.items() if value is None]
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")

    return Stage1ClipTrainConfig(
        **merged,
        train_ids_path=_format_value(file_config.get("train_ids_path"), subject, roi) if subject and roi else file_config.get("train_ids_path"),
        eval_ids_path=_format_value(file_config.get("eval_ids_path", file_config.get("val_ids_path")), subject, roi) if subject and roi else file_config.get("eval_ids_path", file_config.get("val_ids_path")),
        batch_size=file_config.get("batch_size", args.batch_size),
        eval_batch_size=file_config.get("eval_batch_size", args.eval_batch_size),
        num_workers=file_config.get("num_workers", args.num_workers),
        num_epochs=file_config.get("num_epochs", args.num_epochs),
        learning_rate=file_config.get("learning_rate", args.learning_rate),
        weight_decay=file_config.get("weight_decay", args.weight_decay),
        max_grad_norm=file_config.get("max_grad_norm", 1.0),
        lambda_janus=file_config.get("lambda_janus", args.lambda_janus),
        lambda_clip_vision=file_config.get("lambda_clip_vision", args.lambda_clip_vision),
        lambda_clip_text=file_config.get("lambda_clip_text", args.lambda_clip_text),
        hidden_dim=file_config.get("hidden_dim", 4096),
        num_blocks=file_config.get("num_blocks", 4),
        shared_token_count=file_config.get("shared_token_count"),
        shared_token_dim=file_config.get("shared_token_dim"),
        head_hidden_dim=file_config.get("head_hidden_dim", 2048),
        dropout=file_config.get("dropout", 0.1),
        seed=file_config.get("seed", 42),
        mixed_precision=file_config.get("mixed_precision", args.mixed_precision),
        save_every=file_config.get("save_every", 5),
        resume_from=_format_value(file_config.get("resume_from", args.resume_from), subject, roi) if subject and roi else file_config.get("resume_from", args.resume_from),
        subject=subject,
        roi=roi,
    )


def _build_experiment_configs(args: argparse.Namespace, file_config: Dict) -> List[Stage1ClipTrainConfig]:
    subjects = file_config.get("subjects")
    rois = file_config.get("rois")
    if not subjects and not rois:
        return [_build_single_config(args, file_config)]
    if not subjects or not rois:
        raise ValueError("When using batch mode, both 'subjects' and 'rois' must be provided in the config")
    configs: List[Stage1ClipTrainConfig] = []
    for subject in subjects:
        for roi in rois:
            configs.append(_build_single_config(args, file_config, subject=subject, roi=roi))
    return configs


def build_clip_dataloaders(config: ClipMultiTargetDataConfig):
    utils.seed_everything(seed=config.seed)
    train_dataset = ClipMultiTargetDataset(
        fmri_path=config.train_fmri_path,
        janus_vision_path=config.train_janus_vision_path,
        clip_vision_path=config.train_clip_vision_path,
        clip_text_path=config.train_clip_text_path,
        ids_path=config.train_ids_path,
    )
    eval_dataset = ClipMultiTargetDataset(
        fmri_path=config.eval_fmri_path,
        janus_vision_path=config.eval_janus_vision_path,
        clip_vision_path=config.eval_clip_vision_path,
        clip_text_path=config.eval_clip_text_path,
        ids_path=config.eval_ids_path,
    )
    loader_kwargs: Dict = {"num_workers": config.num_workers, "pin_memory": config.pin_memory}
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = config.prefetch_factor
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle_train, drop_last=True, **loader_kwargs)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, eval_loader


# ──────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────

class Stage1ClipTrainer:
    def __init__(self, config: Stage1ClipTrainConfig) -> None:
        self.config = config
        self.accelerator = Accelerator(mixed_precision=config.mixed_precision)

        # Infer token specs from feature files
        janus_probe = np.load(config.train_janus_vision_path, mmap_mode="r")
        clip_vision_probe = np.load(config.train_clip_vision_path, mmap_mode="r")
        clip_text_probe = np.load(config.train_clip_text_path, mmap_mode="r")
        self.janus_spec = TokenSpec(*infer_token_spec(janus_probe))
        self.clip_vision_spec = TokenSpec(*infer_token_spec(clip_vision_probe))
        self.clip_text_spec = TokenSpec(*infer_token_spec(clip_text_probe))

        # Build model — uses the generic Stage1MultiTargetModel; the "siglip" heads
        # now predict CLIP features. The architecture is identical; only target dims differ.
        self.model = build_stage1_model(
            num_voxels=config.num_voxels,
            janus_spec=self.janus_spec,
            siglip_vision_spec=self.clip_vision_spec,   # CLIP vision: (257, 768)
            siglip_text_spec=self.clip_text_spec,        # CLIP text:   (77, 768)
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            shared_token_count=config.shared_token_count,
            shared_token_dim=config.shared_token_dim,
            head_hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        data_config = ClipMultiTargetDataConfig(
            train_fmri_path=config.train_fmri_path,
            train_janus_vision_path=config.train_janus_vision_path,
            train_clip_vision_path=config.train_clip_vision_path,
            train_clip_text_path=config.train_clip_text_path,
            eval_fmri_path=config.eval_fmri_path,
            eval_janus_vision_path=config.eval_janus_vision_path,
            eval_clip_vision_path=config.eval_clip_vision_path,
            eval_clip_text_path=config.eval_clip_text_path,
            train_ids_path=config.train_ids_path,
            eval_ids_path=config.eval_ids_path,
            batch_size=config.batch_size,
            eval_batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
        )
        train_loader, eval_loader = build_clip_dataloaders(data_config)
        total_steps = max(1, len(train_loader) * config.num_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            final_div_factor=1000,
            pct_start=max(0.01, 2 / max(config.num_epochs, 2)),
        )

        self.model, self.optimizer, self.lr_scheduler, self.train_loader, self.eval_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, train_loader, eval_loader,
        )
        os.makedirs(config.output_dir, exist_ok=True)
        self.start_epoch = 0
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def _compute_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        janus_target = batch["janus_vision"].to(self.accelerator.device, non_blocking=True).float()
        clip_vision_target = batch["clip_vision"].to(self.accelerator.device, non_blocking=True).float()
        clip_text_target = batch["clip_text"].to(self.accelerator.device, non_blocking=True).float()

        loss_janus = token_mse_cosine_loss(outputs["janus_visual"], janus_target)
        # model internally calls these heads "siglip_vision/text"; targets are CLIP features
        loss_clip_vision = token_mse_cosine_loss(outputs["siglip_vision"], clip_vision_target)
        loss_clip_text = token_mse_cosine_loss(outputs["siglip_text"], clip_text_target)
        total_loss = (
            self.config.lambda_janus * loss_janus
            + self.config.lambda_clip_vision * loss_clip_vision
            + self.config.lambda_clip_text * loss_clip_text
        )
        return {
            "loss": total_loss,
            "loss_janus": loss_janus,
            "loss_clip_vision": loss_clip_vision,
            "loss_clip_text": loss_clip_text,
        }

    def _step(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, float]:
        voxels = prepare_fmri_batch(batch["fmri"].to(self.accelerator.device, non_blocking=True), train=train)
        autocast_enabled = self.accelerator.device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            outputs = self.model(voxels)
            losses = self._compute_losses(outputs, batch)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(losses["loss"])
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()

        return {key: float(value.detach().item()) for key, value in losses.items()}

    def _run_epoch(self, loader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        aggregate = {"loss": 0.0, "loss_janus": 0.0, "loss_clip_vision": 0.0, "loss_clip_text": 0.0}
        steps = 0
        context = torch.enable_grad() if train else torch.no_grad()
        phase = "train" if train else "eval"
        progress_bar = tqdm(loader, disable=not self.accelerator.is_main_process, leave=False, desc=phase)
        with context:
            for batch in progress_bar:
                metrics = self._step(batch, train=train)
                for key, value in metrics.items():
                    aggregate[key] += value
                steps += 1
                progress_bar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    janus=f"{metrics['loss_janus']:.4f}",
                    clip_v=f"{metrics['loss_clip_vision']:.4f}",
                    clip_t=f"{metrics['loss_clip_text']:.4f}",
                )
        if steps == 0:
            return aggregate
        return {key: value / steps for key, value in aggregate.items()}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filename: str) -> None:
        model = self.accelerator.unwrap_model(self.model)
        torch.save(
            {
                "epoch": epoch,
                "config": asdict(self.config),
                "janus_spec": asdict(self.janus_spec),
                "clip_vision_spec": asdict(self.clip_vision_spec),
                "clip_text_spec": asdict(self.clip_text_spec),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "metrics": metrics,
            },
            filename,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1

    def train(self) -> Dict[str, Dict[str, float]]:
        history: Dict[str, Dict[str, float]] = {}
        best_eval_loss = float("inf")
        progress = tqdm(range(self.start_epoch, self.config.num_epochs), disable=not self.accelerator.is_main_process)
        for epoch in progress:
            train_metrics = self._run_epoch(self.train_loader, train=True)
            eval_metrics = self._run_epoch(self.eval_loader, train=False)
            epoch_metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"eval_{k}": v for k, v in eval_metrics.items()},
            }
            history[f"epoch_{epoch}"] = epoch_metrics
            if self.accelerator.is_main_process:
                progress.set_postfix(
                    train_loss=f"{train_metrics['loss']:.4f}",
                    eval_loss=f"{eval_metrics['loss']:.4f}",
                    janus=f"{eval_metrics['loss_janus']:.4f}",
                )
                self.save_checkpoint(epoch, epoch_metrics, os.path.join(self.config.output_dir, "last_stage1_clip.pth"))
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    self.save_checkpoint(epoch, epoch_metrics, os.path.join(self.config.output_dir, "best_stage1_clip.pth"))
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch, epoch_metrics, os.path.join(self.config.output_dir, f"epoch_{epoch:04d}_stage1_clip.pth"))
            self.accelerator.wait_for_everyone()
        return history


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 multitarget training with CLIP auxiliary supervision")
    parser.add_argument("--config", type=str, default="./config_stage1_multitarget_clip.json")
    parser.add_argument("--train-fmri-path", type=str, default=None)
    parser.add_argument("--train-janus-vision-path", type=str, default=None)
    parser.add_argument("--train-clip-vision-path", type=str, default=None)
    parser.add_argument("--train-clip-text-path", type=str, default=None)
    parser.add_argument("--eval-fmri-path", type=str, default=None)
    parser.add_argument("--eval-janus-vision-path", type=str, default=None)
    parser.add_argument("--eval-clip-vision-path", type=str, default=None)
    parser.add_argument("--eval-clip-text-path", type=str, default=None)
    parser.add_argument("--num-voxels", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-janus", type=float, default=1.0)
    parser.add_argument("--lambda-clip-vision", type=float, default=0.3)
    parser.add_argument("--lambda-clip-text", type=float, default=0.1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_config = _load_config_file(args.config)
    experiment_configs = _build_experiment_configs(args, file_config)
    for index, config in enumerate(experiment_configs, start=1):
        label = f"{config.subject or 'single'}-{config.roi or 'default'}"
        print(f"[{index}/{len(experiment_configs)}] Starting stage1-clip training for {label}")
        print(f"  train_fmri_path   = {config.train_fmri_path}")
        print(f"  clip_vision_path  = {config.train_clip_vision_path}  shape hint: (N,257,768)")
        print(f"  clip_text_path    = {config.train_clip_text_path}  shape hint: (N,77,768)")
        print(f"  output_dir        = {config.output_dir}")
        print(f"  num_voxels        = {config.num_voxels}")

        trainer = Stage1ClipTrainer(config)
        history = trainer.train()
        if trainer.accelerator.is_main_process:
            with open(os.path.join(config.output_dir, "history_stage1_clip.json"), "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

