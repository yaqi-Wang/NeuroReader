from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_paper4 import BrainDiffusionPrior, BrainNetwork, VersatileDiffusionPriorNetwork


@dataclass
class TokenSpec:
    token_count: int
    token_dim: int


def infer_token_spec_from_tensor(tokens: torch.Tensor) -> TokenSpec:
    if tokens.ndim != 3:
        raise ValueError(f"Expected [N, T, D], got {tuple(tokens.shape)}")
    return TokenSpec(token_count=int(tokens.shape[1]), token_dim=int(tokens.shape[2]))


def resample_tokens(tokens: torch.Tensor, target_count: int) -> torch.Tensor:
    if tokens.shape[1] == target_count:
        return tokens
    resized = F.interpolate(tokens.transpose(1, 2), size=target_count, mode="linear", align_corners=False)
    return resized.transpose(1, 2).contiguous()


def token_mse_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mse_weight: float = 1.0, cosine_weight: float = 1.0) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    loss_mse = F.mse_loss(pred, target)
    pred_flat = F.normalize(pred.flatten(1), dim=-1)
    target_flat = F.normalize(target.flatten(1), dim=-1)
    loss_cos = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
    return (mse_weight * loss_mse) + (cosine_weight * loss_cos)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SharedTokenGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        token_count: int,
        token_dim: int,
        token_hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.global_projection = nn.Sequential(
            nn.Linear(hidden_dim, token_hidden_dim),
            nn.LayerNorm(token_hidden_dim),
            nn.GELU(),
        )
        self.scale_shift = nn.Linear(hidden_dim, token_hidden_dim * 2)
        self.query_tokens = nn.Parameter(torch.randn(1, token_count, token_hidden_dim) * (token_hidden_dim ** -0.5))
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_hidden_dim),
            nn.Linear(token_hidden_dim, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, token_hidden_dim),
        )
        self.to_token_dim = nn.Sequential(
            nn.LayerNorm(token_hidden_dim),
            nn.Linear(token_hidden_dim, token_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = hidden.shape[0]
        global_state = self.global_projection(hidden).unsqueeze(1)
        scale, shift = self.scale_shift(hidden).chunk(2, dim=-1)
        tokens = self.query_tokens.expand(batch_size, -1, -1)
        tokens = tokens + global_state
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        tokens = tokens + self.token_mlp(tokens)
        return self.to_token_dim(tokens)


class BrainMultiTargetEncoder(nn.Module):
    def __init__(
        self,
        num_voxels: int,
        shared_token_spec: TokenSpec,
        hidden_dim: int = 4096,
        num_blocks: int = 4,
        token_hidden_dim: int = 512,
        input_dropout: float = 0.5,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.shared_token_spec = shared_token_spec
        self.lin0 = nn.Sequential(
            nn.Linear(num_voxels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(input_dropout),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.token_generator = SharedTokenGenerator(
            hidden_dim=hidden_dim,
            token_count=shared_token_spec.token_count,
            token_dim=shared_token_spec.token_dim,
            token_hidden_dim=token_hidden_dim,
            dropout=dropout,
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        hidden = self.lin0(voxels)
        for block in self.blocks:
            hidden = block(hidden)
        return self.token_generator(hidden)


class TokenPredictionHead(nn.Module):
    def __init__(self, in_dim: int, out_spec: TokenSpec, hidden_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.out_spec = out_spec
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_spec.token_dim),
        )

    def forward(self, shared_tokens: torch.Tensor) -> torch.Tensor:
        tokens = resample_tokens(shared_tokens, self.out_spec.token_count)
        return self.net(tokens)


class JanusVisualHead(TokenPredictionHead):
    pass


class SigLIPVisionHead(TokenPredictionHead):
    pass


class SigLIPTextHead(TokenPredictionHead):
    pass


class TokenProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class Stage1MultiTargetModel(nn.Module):
    def __init__(
        self,
        num_voxels: int,
        janus_spec: TokenSpec,
        siglip_vision_spec: TokenSpec,
        siglip_text_spec: TokenSpec,
        hidden_dim: int = 4096,
        num_blocks: int = 4,
        shared_token_count: Optional[int] = None,
        shared_token_dim: Optional[int] = None,
        head_hidden_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        shared_spec = TokenSpec(
            token_count=shared_token_count or janus_spec.token_count,
            token_dim=shared_token_dim or janus_spec.token_dim,
        )
        self.encoder = BrainMultiTargetEncoder(
            num_voxels=num_voxels,
            shared_token_spec=shared_spec,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.janus_head = JanusVisualHead(shared_spec.token_dim, janus_spec, hidden_dim=head_hidden_dim, dropout=dropout)
        self.siglip_vision_head = SigLIPVisionHead(shared_spec.token_dim, siglip_vision_spec, hidden_dim=head_hidden_dim, dropout=dropout)
        self.siglip_text_head = SigLIPTextHead(shared_spec.token_dim, siglip_text_spec, hidden_dim=head_hidden_dim, dropout=dropout)

    def forward(self, voxels: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_tokens = self.encoder(voxels)
        return {
            "shared_tokens": shared_tokens,
            "janus_visual": self.janus_head(shared_tokens),
            "siglip_vision": self.siglip_vision_head(shared_tokens),
            "siglip_text": self.siglip_text_head(shared_tokens),
        }


class JanusJEPAPredictor(nn.Module):
    def __init__(
        self,
        token_spec: TokenSpec,
        depth: int = 4,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_spec = token_spec
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    nn.LayerNorm(token_spec.token_dim),
                    nn.Linear(token_spec.token_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, token_spec.token_dim),
                ]
            )
        self.net = nn.ModuleList([ResidualMLPBlock(token_spec.token_dim, dropout=dropout) for _ in range(depth)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(token_spec.token_dim),
            nn.Linear(token_spec.token_dim, token_spec.token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        refined = tokens
        for block in self.net:
            refined = block(refined)
        return self.out_proj(refined)


class Stage2JEPAModel(nn.Module):
    def __init__(self, stage1_model: Stage1MultiTargetModel, freeze_stage1: bool = True, predictor_depth: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.stage1_model = stage1_model
        if freeze_stage1:
            self.stage1_model.requires_grad_(False)
        janus_spec = self.stage1_model.janus_head.out_spec
        self.predictor = JanusJEPAPredictor(janus_spec, depth=predictor_depth, dropout=dropout)

    def forward(self, voxels: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.set_grad_enabled(any(param.requires_grad for param in self.stage1_model.parameters())):
            stage1_outputs = self.stage1_model(voxels)
        refined = self.predictor(stage1_outputs["janus_visual"])
        stage1_outputs["janus_refined"] = refined
        return stage1_outputs


class _Stage1VoxelToJanusWrapper(nn.Module):
    def __init__(
        self,
        stage1_model: Stage1MultiTargetModel,
        janus_spec: TokenSpec,
        target_projector: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.stage1_model = stage1_model
        self.janus_spec = janus_spec
        self.use_projector = True
        self.projector = target_projector or nn.Identity()

    def forward(self, voxels: torch.Tensor):
        outputs = self.stage1_model(voxels)
        janus_tokens = outputs["janus_visual"]
        projected_tokens = self.projector(janus_tokens)
        flat = projected_tokens.reshape(projected_tokens.shape[0], -1)
        return flat, projected_tokens


class JanusDiffusionBaseline(nn.Module):
    def __init__(
        self,
        stage1_model: Stage1MultiTargetModel,
        janus_spec: TokenSpec,
        latent_token_dim: int = 1024,
        latent_hidden_dim: int = 2048,
        prior_depth: int = 6,
        prior_timesteps: int = 100,
        learned_query_mode: str = "pos_emb",
        cond_drop_prob: float = 0.2,
        freeze_stage1: bool = True,
    ) -> None:
        super().__init__()
        if freeze_stage1:
            stage1_model.requires_grad_(False)
        self.stage1_model = stage1_model
        self.latent_spec = TokenSpec(token_count=janus_spec.token_count, token_dim=latent_token_dim)
        self.target_projector = TokenProjector(
            in_dim=janus_spec.token_dim,
            out_dim=latent_token_dim,
            hidden_dim=latent_hidden_dim,
            dropout=0.1,
        )
        self.output_projector = TokenProjector(
            in_dim=latent_token_dim,
            out_dim=janus_spec.token_dim,
            hidden_dim=latent_hidden_dim,
            dropout=0.1,
        )
        voxel2clip = _Stage1VoxelToJanusWrapper(stage1_model, janus_spec, target_projector=self.target_projector)
        prior_network = VersatileDiffusionPriorNetwork(
            dim=self.latent_spec.token_dim,
            depth=prior_depth,
            dim_head=64,
            heads=max(1, self.latent_spec.token_dim // 64),
            causal=False,
            num_tokens=self.latent_spec.token_count,
            learned_query_mode=learned_query_mode,
        )
        self.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=self.latent_spec.token_dim,
            condition_on_text_encodings=False,
            timesteps=prior_timesteps,
            cond_drop_prob=cond_drop_prob,
            image_embed_scale=None,
            voxel2clip=voxel2clip,
        )
        self.janus_spec = janus_spec

    def forward(self, voxels: torch.Tensor, janus_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        janus_target_latent = self.target_projector(janus_target)
        loss_prior, pred_latent = self.diffusion_prior(voxel=voxels, image_embed=janus_target_latent)
        stage1_outputs = self.stage1_model(voxels)
        stage1_outputs["janus_target_latent"] = janus_target_latent
        stage1_outputs["janus_refined_latent"] = pred_latent
        stage1_outputs["janus_refined"] = self.output_projector(pred_latent)
        stage1_outputs["loss_prior"] = loss_prior
        return stage1_outputs

    @torch.no_grad()
    def sample(self, voxels: torch.Tensor, generator: Optional[torch.Generator] = None, timesteps: Optional[int] = None) -> torch.Tensor:
        _, init_tokens = self.diffusion_prior.voxel2clip(voxels)
        sampled_latent = self.diffusion_prior.p_sample_loop(
            init_tokens.shape,
            text_cond=dict(text_embed=init_tokens),
            cond_scale=1.0,
            timesteps=timesteps or self.diffusion_prior.noise_scheduler.num_timesteps,
            generator=generator,
        )
        return self.output_projector(sampled_latent)


def build_stage1_model(
    num_voxels: int,
    janus_spec: TokenSpec,
    siglip_vision_spec: TokenSpec,
    siglip_text_spec: TokenSpec,
    hidden_dim: int = 4096,
    num_blocks: int = 4,
    shared_token_count: Optional[int] = None,
    shared_token_dim: Optional[int] = None,
    head_hidden_dim: int = 2048,
    dropout: float = 0.1,
) -> Stage1MultiTargetModel:
    return Stage1MultiTargetModel(
        num_voxels=num_voxels,
        janus_spec=janus_spec,
        siglip_vision_spec=siglip_vision_spec,
        siglip_text_spec=siglip_text_spec,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        shared_token_count=shared_token_count,
        shared_token_dim=shared_token_dim,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
    )
