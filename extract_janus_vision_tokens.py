import argparse
import inspect
import json
import os
from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from nsd_access import NSDAccess


def _collect_prepare_inputs_kwargs(prepare_inputs: Any) -> Dict[str, Any]:
    keys = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "images_seq_mask",
        "images_emb_mask",
        "sft_format",
    ]
    kwargs: Dict[str, Any] = {}
    for key in keys:
        value = getattr(prepare_inputs, key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _call_prepare_inputs_embeds(
    vl_gpt: MultiModalityCausalLM,
    prepare_inputs: Any,
    pixel_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kwargs = _collect_prepare_inputs_kwargs(prepare_inputs)
    method = vl_gpt.prepare_inputs_embeds

    call_variants = []
    if kwargs:
        call_variants.append(("kwargs", lambda: method(**kwargs)))
    if pixel_values is not None:
        call_variants.append(("pixel_values_only", lambda: method(pixel_values)))
        call_variants.append(("pixel_values_kw", lambda: method(pixel_values=pixel_values)))
        input_ids = kwargs.get("input_ids")
        images_seq_mask = kwargs.get("images_seq_mask")
        images_emb_mask = kwargs.get("images_emb_mask")
        if input_ids is not None and images_seq_mask is not None and images_emb_mask is not None:
            call_variants.append(
                (
                    "pixel_plus_masks",
                    lambda: method(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        images_seq_mask=images_seq_mask,
                        images_emb_mask=images_emb_mask,
                    ),
                )
            )

    errors = []
    for name, fn in call_variants:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - best effort fallback path
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    signature = str(inspect.signature(method))
    joined_errors = "\n".join(errors)
    raise RuntimeError(
        "Unable to call vl_gpt.prepare_inputs_embeds with the current Janus API.\n"
        f"Detected signature: {signature}\n"
        f"Tried variants:\n{joined_errors}"
    )


def _extract_image_slot_embeddings(inputs_embeds: torch.Tensor, images_seq_mask: torch.Tensor) -> torch.Tensor:
    if images_seq_mask is None:
        raise ValueError("Janus processor output is missing images_seq_mask")
    images_seq_mask = images_seq_mask.bool()
    batch_size = inputs_embeds.shape[0]
    per_batch = images_seq_mask.sum(dim=1)
    if int(per_batch.min().item()) != int(per_batch.max().item()):
        raise ValueError("Image token counts differ across the batch; this script expects a fixed image token count")
    token_count = int(per_batch[0].item())
    hidden_dim = int(inputs_embeds.shape[-1])
    return inputs_embeds[images_seq_mask].view(batch_size, token_count, hidden_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Janus visual token targets from NSD images")
    parser.add_argument("--imgidx", default=[56220, 73000], nargs="*", type=int, help="start and end image indices")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--janus_model_path",
        type=str,
        default="/home/data/wangyaqi/projects/18BrainGPT/DeepSeek",
        help="Local path or HF id for Janus model",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main",
        # default="/home/janus_vision",
        help="Root path for saving extracted Janus features",
    )
    parser.add_argument(
        "--nsd_data_path",
        type=str,
        default="/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/",
        help="Path to NSD data root",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Provide a general description of the perceived scene.",
        help="Prompt used to construct the image placeholder conversation",
    )
    parser.add_argument(
        "--use_autocast",
        action="store_true",
        help="Enable torch autocast on CUDA",
    )
    opt = parser.parse_args()

    if len(opt.imgidx) != 2:
        raise ValueError("--imgidx must provide exactly 2 ints: start end")

    seed_everything(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        device = torch.device(f"cuda:{opt.gpu}")
    else:
        device = torch.device("cpu")

    nsda = NSDAccess(opt.nsd_data_path)
    output_dir = os.path.join(opt.root_path, "nsdfeat", "janus_vision")
    os.makedirs(output_dir, exist_ok=True)

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(opt.janus_model_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        opt.janus_model_path,
        trust_remote_code=True,
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()

    if opt.use_autocast and device.type == "cuda":
        precision_scope = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        precision_scope = nullcontext

    start_idx, end_idx = opt.imgidx
    feature_shape = None

    for image_index in tqdm(range(start_idx, end_idx)):
        img_arr = nsda.read_images(image_index)
        pil_image = Image.fromarray(img_arr).convert("RGB")
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{opt.question}",
                "images": [pil_image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = vl_chat_processor(conversations=conversation, images=[pil_image], force_batchify=True).to(device)
        pixel_values = getattr(prepare_inputs, "pixel_values", None)

        with torch.no_grad():
            with precision_scope():
                inputs_embeds = _call_prepare_inputs_embeds(vl_gpt, prepare_inputs, pixel_values=pixel_values)
                image_slot_embeds = _extract_image_slot_embeddings(inputs_embeds, getattr(prepare_inputs, "images_seq_mask", None))

        image_slot_embeds = image_slot_embeds.squeeze(0).detach().cpu().to(torch.float32).numpy()
        np.save(os.path.join(output_dir, f"{image_index:06}.npy"), image_slot_embeds)

        if feature_shape is None:
            feature_shape = list(image_slot_embeds.shape)

    if feature_shape is not None: 
        metadata = {
            "question": opt.question,
            "feature_shape": feature_shape,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "janus_model_path": opt.janus_model_path,
            "notes": "Extracted image-slot embeddings from Janus prepare_inputs_embeds output",
        }
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
