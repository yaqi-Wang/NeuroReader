import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from data_multitarget import MultiTargetDataConfig, build_multitarget_dataloaders, prepare_fmri_batch
from models_multitarget import Stage2JEPAModel, TokenSpec, build_stage1_model, token_mse_cosine_loss


@dataclass
class Stage2JEPAConfig:
    stage1_checkpoint_path: str
    train_fmri_path: str
    train_janus_vision_path: str
    train_siglip_vision_path: str
    train_siglip_text_path: str
    eval_fmri_path: str
    eval_janus_vision_path: str
    eval_siglip_vision_path: str
    eval_siglip_text_path: str
    output_dir: str
    train_ids_path: Optional[str] = None
    eval_ids_path: Optional[str] = None
    batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 0
    num_epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    freeze_stage1: bool = True
    predictor_depth: int = 4
    lambda_janus: float = 1.0
    lambda_consistency: float = 0.1
    mixed_precision: str = "bf16"
    seed: int = 42
    save_every: int = 5
    resume_from: Optional[str] = None
    subject: Optional[str] = None
    roi: Optional[str] = None


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


def _build_single_config(args: argparse.Namespace, file_config: Dict, subject: Optional[str] = None, roi: Optional[str] = None) -> Stage2JEPAConfig:
    required = {
        "stage1_checkpoint_path": file_config.get("stage1_checkpoint_path", args.stage1_checkpoint_path),
        "train_fmri_path": file_config.get("train_fmri_path", args.train_fmri_path),
        "train_janus_vision_path": file_config.get("train_janus_vision_path", args.train_janus_vision_path),
        "train_siglip_vision_path": file_config.get("train_siglip_vision_path", args.train_siglip_vision_path),
        "train_siglip_text_path": file_config.get("train_siglip_text_path", args.train_siglip_text_path),
        "eval_fmri_path": file_config.get("eval_fmri_path", file_config.get("val_fmri_path", args.eval_fmri_path)),
        "eval_janus_vision_path": file_config.get("eval_janus_vision_path", file_config.get("val_janus_vision_path", args.eval_janus_vision_path)),
        "eval_siglip_vision_path": file_config.get("eval_siglip_vision_path", file_config.get("val_siglip_vision_path", args.eval_siglip_vision_path)),
        "eval_siglip_text_path": file_config.get("eval_siglip_text_path", file_config.get("val_siglip_text_path", args.eval_siglip_text_path)),
        "output_dir": file_config.get("output_dir", args.output_dir),
    }
    if subject is not None and roi is not None:
        required = {key: _format_value(value, subject, roi) for key, value in required.items()}

    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")

    return Stage2JEPAConfig(
        **required,
        train_ids_path=_format_value(file_config.get("train_ids_path"), subject, roi) if subject and roi else file_config.get("train_ids_path"),
        eval_ids_path=_format_value(file_config.get("eval_ids_path", file_config.get("val_ids_path")), subject, roi) if subject and roi else file_config.get("eval_ids_path", file_config.get("val_ids_path")),
        batch_size=file_config.get("batch_size", 4),
        eval_batch_size=file_config.get("eval_batch_size", 4),
        num_workers=file_config.get("num_workers", 0),
        num_epochs=file_config.get("num_epochs", 40),
        learning_rate=file_config.get("learning_rate", 1e-4),
        weight_decay=file_config.get("weight_decay", 1e-2),
        max_grad_norm=file_config.get("max_grad_norm", 1.0),
        freeze_stage1=file_config.get("freeze_stage1", True),
        predictor_depth=file_config.get("predictor_depth", 4),
        lambda_janus=file_config.get("lambda_janus", 1.0),
        lambda_consistency=file_config.get("lambda_consistency", 0.1),
        mixed_precision=file_config.get("mixed_precision", "bf16"),
        seed=file_config.get("seed", 42),
        save_every=file_config.get("save_every", 5),
        resume_from=_format_value(file_config.get("resume_from"), subject, roi) if subject and roi else file_config.get("resume_from"),
        subject=subject,
        roi=roi,
    )


def _build_experiment_configs(args: argparse.Namespace, file_config: Dict) -> List[Stage2JEPAConfig]:
    subjects = file_config.get("subjects")
    rois = file_config.get("rois")
    if not subjects and not rois:
        return [_build_single_config(args, file_config)]
    if not subjects or not rois:
        raise ValueError("When using batch mode, both 'subjects' and 'rois' must be provided in the config")
    configs: List[Stage2JEPAConfig] = []
    for subject in subjects:
        for roi in rois:
            configs.append(_build_single_config(args, file_config, subject=subject, roi=roi))
    return configs


def _load_stage1_model(stage1_checkpoint_path: str):
    checkpoint = torch.load(stage1_checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    janus_spec = TokenSpec(**checkpoint["janus_spec"])
    siglip_vision_spec = TokenSpec(**checkpoint["siglip_vision_spec"])
    siglip_text_spec = TokenSpec(**checkpoint["siglip_text_spec"])
    model = build_stage1_model(
        num_voxels=config["num_voxels"],
        janus_spec=janus_spec,
        siglip_vision_spec=siglip_vision_spec,
        siglip_text_spec=siglip_text_spec,
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        shared_token_count=config.get("shared_token_count"),
        shared_token_dim=config.get("shared_token_dim"),
        head_hidden_dim=config.get("head_hidden_dim", 2048),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model, checkpoint


class Stage2JEPATrainer:
    def __init__(self, config: Stage2JEPAConfig) -> None:
        self.config = config
        self.accelerator = Accelerator(mixed_precision=config.mixed_precision)
        stage1_model, stage1_checkpoint = _load_stage1_model(config.stage1_checkpoint_path)
        self.stage1_checkpoint = stage1_checkpoint
        self.model = Stage2JEPAModel(
            stage1_model=stage1_model,
            freeze_stage1=config.freeze_stage1,
            predictor_depth=config.predictor_depth,
        )
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

        data_config = MultiTargetDataConfig(
            train_fmri_path=config.train_fmri_path,
            train_janus_vision_path=config.train_janus_vision_path,
            train_siglip_vision_path=config.train_siglip_vision_path,
            train_siglip_text_path=config.train_siglip_text_path,
            eval_fmri_path=config.eval_fmri_path,
            eval_janus_vision_path=config.eval_janus_vision_path,
            eval_siglip_vision_path=config.eval_siglip_vision_path,
            eval_siglip_text_path=config.eval_siglip_text_path,
            train_ids_path=config.train_ids_path,
            eval_ids_path=config.eval_ids_path,
            batch_size=config.batch_size,
            eval_batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
        )
        train_loader, eval_loader = build_multitarget_dataloaders(data_config)
        total_steps = max(1, len(train_loader) * config.num_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            final_div_factor=1000,
            pct_start=max(0.01, 2 / max(config.num_epochs, 2)),
        )
        self.model, self.optimizer, self.lr_scheduler, self.train_loader, self.eval_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            train_loader,
            eval_loader,
        )
        os.makedirs(config.output_dir, exist_ok=True)
        self.start_epoch = 0
        self.best_eval_loss = float("inf")
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def _step(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, float]:
        voxels = prepare_fmri_batch(batch["fmri"].to(self.accelerator.device, non_blocking=True), train=train)
        janus_target = batch["janus_vision"].to(self.accelerator.device, non_blocking=True).float()
        with torch.amp.autocast(device_type="cuda", enabled=self.accelerator.device.type == "cuda"):
            outputs = self.model(voxels)
            loss_janus = token_mse_cosine_loss(outputs["janus_refined"], janus_target)
            loss_consistency = token_mse_cosine_loss(outputs["janus_refined"], outputs["janus_visual"])
            total_loss = self.config.lambda_janus * loss_janus + self.config.lambda_consistency * loss_consistency
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(total_loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
        return {
            "loss": float(total_loss.detach().item()),
            "loss_janus": float(loss_janus.detach().item()),
            "loss_consistency": float(loss_consistency.detach().item()),
        }

    def _run_epoch(self, loader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        aggregate = {"loss": 0.0, "loss_janus": 0.0, "loss_consistency": 0.0}
        steps = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                metrics = self._step(batch, train=train)
                for key, value in metrics.items():
                    aggregate[key] += value
                steps += 1
        if steps == 0:
            return aggregate
        return {key: value / steps for key, value in aggregate.items()}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filename: str) -> None:
        model = self.accelerator.unwrap_model(self.model)
        torch.save(
            {
                "epoch": epoch,
                "config": asdict(self.config),
                "stage1_checkpoint_path": self.config.stage1_checkpoint_path,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "best_eval_loss": self.best_eval_loss,
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
        self.best_eval_loss = float(checkpoint.get("best_eval_loss", checkpoint.get("metrics", {}).get("eval_loss", float("inf"))))

    def train(self) -> Dict[str, Dict[str, float]]:
        history: Dict[str, Dict[str, float]] = {}
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
                progress.set_postfix(train_loss=f"{train_metrics['loss']:.4f}", eval_loss=f"{eval_metrics['loss']:.4f}")
                self.save_checkpoint(epoch, epoch_metrics, os.path.join(self.config.output_dir, "last_stage2_jepa.pth"))
                if eval_metrics["loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["loss"]
                    self.save_checkpoint(epoch, epoch_metrics, os.path.join(self.config.output_dir, "best_stage2_jepa.pth"))
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch, epoch_metrics, os.path.join(self.config.output_dir, f"epoch_{epoch:04d}_stage2_jepa.pth"))
            self.accelerator.wait_for_everyone()
        return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 JEPA refinement for Janus visual targets")
    parser.add_argument("--config", type=str, default="./config_stage2_jepa.json")
    parser.add_argument("--stage1-checkpoint-path", type=str, default=None)
    parser.add_argument("--train-fmri-path", type=str, default=None)
    parser.add_argument("--train-janus-vision-path", type=str, default=None)
    parser.add_argument("--train-siglip-vision-path", type=str, default=None)
    parser.add_argument("--train-siglip-text-path", type=str, default=None)
    parser.add_argument("--eval-fmri-path", type=str, default=None)
    parser.add_argument("--eval-janus-vision-path", type=str, default=None)
    parser.add_argument("--eval-siglip-vision-path", type=str, default=None)
    parser.add_argument("--eval-siglip-text-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_config = _load_config_file(args.config)
    experiment_configs = _build_experiment_configs(args, file_config)
    for index, config in enumerate(experiment_configs, start=1):
        label = f"{config.subject or 'single'}-{config.roi or 'default'}"
        print(f"[{index}/{len(experiment_configs)}] Starting stage2 JEPA training for {label}")
        print(f"stage1_checkpoint_path={config.stage1_checkpoint_path}")
        print(f"train_fmri_path={config.train_fmri_path}")
        print(f"eval_fmri_path={config.eval_fmri_path}")
        print(f"output_dir={config.output_dir}")

        trainer = Stage2JEPATrainer(config)
        history = trainer.train()
        if trainer.accelerator.is_main_process:
            with open(os.path.join(config.output_dir, "history_stage2_jepa.json"), "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
