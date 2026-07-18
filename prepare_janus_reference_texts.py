import argparse
import csv
import gc
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.io
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from inference_multistage import _call_prepare_inputs_embeds
from janus.models import MultiModalityCausalLM, VLChatProcessor
from nsd_access import NSDAccess


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
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


def _format_value(value: Any, subject: Optional[str], question_key: Optional[str] = None) -> Any:
    if not isinstance(value, str) or subject is None:
        return value
    subject_num = _subject_to_num(subject)
    return value.format(
        subject=subject,
        subject_num=subject_num,
        subject_num_2d=f"{subject_num:02d}",
        sub=f"sub{subject_num}",
        question_key=question_key or "",
    )


def _load_subject_test_indices(nsd_data_path: str, stims_ave_path: str) -> Tuple[np.ndarray, np.ndarray]:
    nsd_expdesign = scipy.io.loadmat(f"{nsd_data_path}/nsddata/experiments/nsd/nsd_expdesign.mat")
    sharedix = nsd_expdesign["sharedix"] - 1
    stims_ave = np.load(stims_ave_path)

    tr_idx = np.zeros_like(stims_ave)
    shared_set = set(sharedix.flatten().tolist())
    for idx, stim_id in enumerate(stims_ave.tolist()):
        tr_idx[idx] = 0 if stim_id in shared_set else 1
    test_indices = np.where(tr_idx == 0)[0]
    return stims_ave, test_indices


@dataclass
class ReferenceRun:
    subject: str
    question_key: str
    question_text: str
    output_root: str


class JanusImageReferenceEngine:
    def __init__(self, janus_model_path: str, device: str = "cuda", max_new_tokens: int = 256) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.max_new_tokens = max_new_tokens
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(janus_model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            janus_model_path,
            trust_remote_code=True,
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).to(self.device).eval()

    @torch.no_grad()
    def generate(self, pil_image: Image.Image, question: str) -> str:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [pil_image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=[pil_image], force_batchify=True).to(self.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            inputs_embeds, _ = _call_prepare_inputs_embeds(self.vl_gpt, prepare_inputs, janus_tokens=None)
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        text = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        del prepare_inputs, inputs_embeds, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text


def _release_engine(engine: Optional[JanusImageReferenceEngine]) -> None:
    if engine is None:
        return
    try:
        if getattr(engine, "vl_gpt", None) is not None:
            engine.vl_gpt.to("cpu")
            engine.vl_gpt = None
    except Exception:
        pass
    del engine 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Janus-Pro reference texts from real NSD images")
    parser.add_argument("--config", type=str, default='configs/config_prepare_janus_reference_texts.json')
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--question-key", type=str, default=None)
    parser.add_argument("--janus-model-path", type=str, default=None)
    parser.add_argument("--nsd-data-path", type=str, default=None)
    parser.add_argument("--root-path", type=str, default=None)
    parser.add_argument("--stims-ave-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--imgidx", nargs=2, type=int, metavar=("START", "END"), default=None)
    return parser.parse_args()


def _build_runs(args: argparse.Namespace, file_config: Dict[str, Any]) -> List[ReferenceRun]:
    subjects = [args.subject] if args.subject else file_config.get("subjects") or ([file_config["subject"]] if file_config.get("subject") else None)
    if not subjects:
        raise ValueError("subject or subjects must be provided")

    questions = file_config.get("questions") or {}
    if args.question_key:
        if args.question_key not in questions:
            raise ValueError(f"Unknown question key '{args.question_key}'. Available keys: {', '.join(questions.keys())}")
        question_map = {args.question_key: questions[args.question_key]}
    else:
        question_keys = file_config.get("question_keys") or ["question1", "question2", "question3"]
        missing = [key for key in question_keys if key not in questions]
        if missing:
            raise ValueError(f"Unknown question keys in config: {', '.join(missing)}")
        question_map = {key: questions[key] for key in question_keys}

    output_path_template = args.output_path or file_config.get("output_path")
    if not output_path_template:
        raise ValueError("output_path must be provided in config or via CLI")

    runs: List[ReferenceRun] = []
    for subject in subjects:
        for question_key, question_text in question_map.items():
            runs.append(
                ReferenceRun(
                    subject=subject,
                    question_key=question_key,
                    question_text=question_text,
                    output_root=_format_value(output_path_template, subject, question_key),
                )
            )
    return runs


def main() -> None:
    args = parse_args()
    file_config = _load_config_file(args.config)
    runs = _build_runs(args, file_config)

    janus_model_path = args.janus_model_path or file_config.get("janus_model_path")
    nsd_data_path = args.nsd_data_path or file_config.get("nsd_data_path")
    root_path = args.root_path or file_config.get("root_path")
    device = args.device or file_config.get("device", "cuda")
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else file_config.get("max_new_tokens", 256)
    imgidx = args.imgidx or file_config.get("imgidx") or [0, 981]
    if len(imgidx) != 2:
        raise ValueError("imgidx must contain exactly [START, END)")
    if not janus_model_path or not nsd_data_path or not root_path:
        raise ValueError("janus_model_path, nsd_data_path, and root_path must be provided")

    active_engine: Optional[JanusImageReferenceEngine] = None
    active_subject: Optional[str] = None
    sf: Optional[h5py.File] = None
    sdataset = None
    stims_ave = None
    test_indices = None

    for run_index, run in enumerate(runs, start=1):
        print(f"[{run_index}/{len(runs)}] Generating Janus references for {run.subject}-{run.question_key}")
        if active_engine is None:
            active_engine = JanusImageReferenceEngine(janus_model_path=janus_model_path, device=device, max_new_tokens=max_new_tokens)

        if active_subject != run.subject:
            if sf is not None:
                sf.close()
            stims_ave_path = args.stims_ave_path or _format_value(
                file_config.get("stims_ave_path") or f"{root_path}/mrifeat_0526/{{subject}}/{{subject}}_stims_ave.npy",
                run.subject,
            )
            stims_ave, test_indices = _load_subject_test_indices(nsd_data_path, stims_ave_path)
            nsda = NSDAccess(nsd_data_path)
            sf = h5py.File(nsda.stimuli_file, "r")
            sdataset = sf.get("imgBrick")
            if sdataset is None:
                raise ValueError(f"Unable to locate imgBrick in {nsda.stimuli_file}")
            active_subject = run.subject

        assert stims_ave is not None and test_indices is not None and sdataset is not None
        sel_start = max(0, int(imgidx[0]))
        sel_end = min(int(imgidx[1]), len(test_indices))
        if sel_start >= sel_end:
            raise ValueError(f"Invalid imgidx range {imgidx} for subject {run.subject}; test count={len(test_indices)}")

        output_dir = run.output_root
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, f"{run.question_key}.csv")
        output_json_path = os.path.join(output_dir, f"{run.question_key}.json")

        rows: List[Dict[str, Any]] = []
        for current_selection_idx in tqdm(range(sel_start, sel_end), desc=f"{run.subject}-{run.question_key}"):
            single_imgidx_te = int(test_indices[current_selection_idx])
            single_idx73k = int(stims_ave[single_imgidx_te])
            img_arr = sdataset[single_idx73k, :, :, :]
            pil_image = Image.fromarray(img_arr).convert("RGB")
            text = active_engine.generate(pil_image, run.question_text)
            rows.append(
                {
                    "sample_index": int(current_selection_idx),
                    "subject_internal_index": single_imgidx_te,
                    "nsd_image_index": single_idx73k,
                    "question_key": run.question_key,
                    "question": run.question_text,
                    "text": text,
                }
            )

        with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_index", "text"],
            )
            writer.writeheader()
            writer.writerows(
                {
                    "sample_index": row["sample_index"],
                    "text": row["text"],
                }
                for row in rows
            )

        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "subject": run.subject,
                    "question_key": run.question_key,
                    "question": run.question_text,
                    "selection_range": [sel_start, sel_end],
                    "num_samples": len(rows),
                    "rows": rows,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    if sf is not None:
        sf.close()
    if active_engine is not None:
        _release_engine(active_engine)


if __name__ == "__main__":
    main()
