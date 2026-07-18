# NeuroReader: Prompt-Conditioned Brain-to-Language Decoding from fMRI

NeuroReader is a multi-stage brain-to-language framework that maps functional magnetic resonance imaging (fMRI) responses to prompt-conditioned natural-language descriptions. Instead of reconstructing pixels, it predicts continuous visual representations compatible with a large vision-language model and uses them to answer questions about scene content, objects, relations, activities, interactions, and contextual cues. We will upload the complete code after receiving the article.

## Graphical abstract

<p align="center">
  <img src="Figure%20abstruct.png" alt="Graphical abstract of NeuroReader" width="100%">
</p>

## Overview

NeuroReader separates brain representation learning from generative-interface calibration:

1. **Stage 1 — multi-target representation alignment.** An fMRI encoder produces shared semantic tokens. Three prediction heads align these tokens with the Janus visual-token interface, SigLIP visual representations, and SigLIP text representations using weighted MSE and cosine losses.
2. **Stage 2 — latent representation refinement (LRR).** A lightweight residual predictor refines the Stage-1 Janus tokens and reduces their distribution mismatch with native Janus visual tokens. A diffusion-prior implementation is included as a comparison baseline.
3. **Prompt-conditioned generation.** The refined brain-derived tokens replace the image-token embeddings in Janus-Pro-7B, allowing the same fMRI response to be queried with different natural-language prompts.

The code also includes feature retrieval, noise robustness, structural ablation, complexity profiling, caption evaluation, and high-level semantic evaluation.

## Repository structure

```text
.
├── configs/                              # Training, inference, and evaluation configurations
├── data/                                 # Caption and semantic-answer references
├── inference_outputs/                    # Generated text from existing experiments
├── nsd_access/                           # Natural Scenes Dataset access utilities
├── results/                              # Saved evaluation summaries
├── Figure abstruct.png                   # Graphical abstract used above
├── data_multitarget.py                   # Dataset and fMRI repeat handling
├── models_multitarget.py                 # Stage 1, LRR, and diffusion-prior models
├── models_paper4.py                      # Diffusion-prior components
├── train_stage1_multitarget.py           # Main Stage-1 training entry point
├── train_stage1_multitarget_clip.py      # CLIP auxiliary-target ablation
├── train_stage1_multitarget_evaclip.py   # EVA-CLIP auxiliary-target ablation
├── train_stage2_jepa.py                  # Main Stage-2 LRR training entry point
├── train_stage2_diffusion_baseline.py    # Diffusion-prior comparison
├── inference_multistage.py               # Single-sample inference
├── inference_multistage_samples.py       # Batch inference
├── extract_janus_vision_tokens.py        # Janus token extraction from NSD images
├── prepare_caption_references.py         # COCO caption-reference preparation
├── prepare_janus_reference_texts.py      # Reference answers from ground-truth images
├── export_predicted_janus_tokens.py      # Token export for retrieval/robustness analysis
├── evaluate_token_retrieval.py           # Feature-retrieval metrics
├── evaluate_caption_metrics.py           # METEOR, ROUGE, and CLIP-Text metrics
├── evaluate_semantic_bertscore-all.py    # Full BERTScore evaluation
├── evaluate_semantic_judge_Bert.py       # Sentence-BERT semantic similarity
├── evaluate_semantic_judge-all.py        # Sentence-BERT plus Janus LLM judging
├── profile_model.py                      # Complexity and speed profiling
├── run_ablation_stage1.sh                # Stage-1 ablations
├── run_ablation_stage2.sh                # Stage-2 ablations
└── run_noise_robustness.sh                # Noise robustness experiments
```

## Requirements

The experiments were run on Linux with a single NVIDIA RTX A5000 GPU (24 GB). A CUDA-enabled PyTorch installation is recommended; the training configurations use bfloat16 mixed precision by default.

Core dependencies used by the repository include:

- PyTorch, torchvision, NumPy, SciPy, pandas, h5py, Pillow, and tqdm
- Hugging Face Transformers and Accelerate
- Janus / Janus-Pro-7B
- OpenCLIP, OpenAI CLIP, and `dalle2-pytorch`
- Sentence Transformers, BERTScore, NLTK, and `rouge-score`
- PyTorch Lightning, WebDataset, matplotlib, and fvcore (for profiling)

Install PyTorch for the CUDA version available on your machine, then install the remaining packages in a clean environment. The Janus installation must expose `janus.models.MultiModalityCausalLM` and `janus.models.VLChatProcessor`. Download the Janus-Pro-7B weights separately and set `janus_model_path` in the relevant configuration files.

> **Note:** The current release does not include a pinned environment file. Record package versions before attempting exact numerical reproduction.

## Data preparation

This project uses subject-specific fMRI responses from the Natural Scenes Dataset (NSD). Access to NSD data is governed by the dataset provider's terms.

### Expected fMRI arrays

Each subject–ROI pair requires training and test arrays in NumPy format:

```text
sub{subject_num}_nsd_train_{roi}_fmriavg.npy
sub{subject_num}_nsd_test_{roi}_fmriavg.npy
```

The last dimension is the number of voxels. When repeated fMRI measurements are available, training uses stochastic repeat aggregation, while evaluation uses their arithmetic mean.

### Expected target features

Stage 1 expects precomputed, subject-aligned feature arrays:

```text
{subject}_ave_janus_vision_tr.npy
{subject}_ave_janus_vision_te.npy
{subject}_ave_siglip_vision_tr.npy
{subject}_ave_siglip_vision_te.npy
{subject}_ave_siglip_text_tr.npy
{subject}_ave_siglip_text_te.npy
```

Token counts and dimensions are inferred at runtime. `extract_janus_vision_tokens.py` extracts per-image Janus tokens from NSD images. The current repository does **not** include the complete subject-level aggregation pipeline or the SigLIP feature-extraction pipeline; prepare these upstream features before Stage-1 training.

### Reference texts

Prepare COCO captions aligned with the subject-specific NSD test order:

```bash
python prepare_caption_references.py \
  --subject subj01 \
  --imgidx 0 982 \
  --output-path data/reference_texts/caption_refs_0_981.npy \
  --output-json-path data/reference_texts/caption_refs_0_981.json
```

Generate prompt-specific reference answers from the ground-truth NSD images:

```bash
python prepare_janus_reference_texts.py \
  --config configs/config_prepare_janus_reference_texts.json
```

## Configuration

All major workflows are driven by JSON files in `configs/`. The committed files are experiment snapshots and contain machine-specific absolute paths. Update every data, checkpoint, model, and output path before running the code.

| Configuration | Purpose |
| --- | --- |
| `config_stage1_multitarget.json` | Main Janus + SigLIP Stage-1 training |
| `config_stage1_multitarget_clip.json` | CLIP auxiliary-target ablation |
| `config_stage1_multitarget_evaclip.json` | EVA-CLIP auxiliary-target ablation |
| `config_stage2_jepa.json` | Main deterministic LRR training |
| `config_stage2_diffusion_baseline.json` | Diffusion-prior comparison |
| `config_inference_multistage_samples.json` | Batch text generation |
| `config_export_predicted_janus_tokens.json` | Token export and control modes |
| `config_evaluate_token_retrieval.json` | Feature-retrieval evaluation |
| `config_evaluate_caption_metrics.json` | Captioning evaluation |
| `config_evaluate_semantic_bertscore.json` | BERTScore evaluation |
| `config_evaluate_semantic_judge.json` | MPNet similarity and optional LLM judge |

For every Stage-2 subject–ROI pair, a matching Stage-1 checkpoint must already exist. For formal `jepa` or `diffusion` inference, always provide the corresponding trained Stage-2 checkpoint; otherwise, the code can instantiate an untrained refinement module.

## Training

### Stage 1: multi-target alignment

```bash
python train_stage1_multitarget.py \
  --config configs/config_stage1_multitarget.json
```

The script trains one model per subject–ROI pair and writes best, last, and periodic checkpoints together with a JSON training history.

### Stage 2: latent representation refinement

Train the main deterministic refinement model:

```bash
python train_stage2_jepa.py \
  --config configs/config_stage2_jepa.json
```

Train the diffusion-prior baseline:

```bash
python train_stage2_diffusion_baseline.py \
  --config configs/config_stage2_diffusion_baseline.json
```

Default experiment settings use 40 Stage-1 epochs with batch size 128 and 30 Stage-2 epochs with batch size 64. Both stages use AdamW, OneCycleLR, gradient clipping, bfloat16 mixed precision, and seed 42.

## Inference

### Single-sample inference

Run a smoke test before batch generation:

```bash
python inference_multistage.py \
  --mode jepa \
  --stage1-checkpoint-path outputs_stage1_multitarget/subj01/nsdgeneral/best_stage1_multitarget.pth \
  --stage2-checkpoint-path outputs_stage2_jepa/subj01/nsdgeneral/best_stage2_jepa.pth \
  --janus-model-path /path/to/Janus-Pro-7B \
  --fmri-path /path/to/sub1_nsd_test_nsdgeneral_fmriavg.npy \
  --num-voxels <VOXEL_COUNT> \
  --sample-index 0 \
  --question "Provide a general description of the perceived scene." \
  --max-new-tokens 64 \
  --save-path inference_outputs/smoke/subj01_nsdgeneral_question1.txt
```

Supported modes are `stage1`, `jepa`, and `diffusion`.

### Batch inference

Set `mode`, subjects, ROIs, questions, paths, and sample range in `configs/config_inference_multistage_samples.json`, then run:

```bash
python inference_multistage_samples.py \
  --config configs/config_inference_multistage_samples.json
```

Each run produces:

```text
inference_outputs/{subject}/{roi}/{question_key}/
├── generated_texts.csv
└── run_config.json
```

`generated_texts.csv` contains `sample_index,text` and supports continuation from an existing file.

> **Index convention:** batch inference treats `[START, END]` as inclusive (for example, `[0, 981]` gives 982 samples). Reference preparation, token export, and retrieval evaluation use `[START, END)` (for example, `[0, 982]` gives 982 samples).

## Evaluation

### Caption metrics

```bash
python evaluate_caption_metrics.py \
  --config configs/config_evaluate_caption_metrics.json
```

Reports METEOR, ROUGE-1, ROUGE-L, and CLIP-Text similarity against aligned COCO captions.

### High-level semantic metrics

```bash
python evaluate_semantic_bertscore-all.py \
  --config configs/config_evaluate_semantic_bertscore.json

python evaluate_semantic_judge_Bert.py \
  --config configs/config_evaluate_semantic_judge.json \
  --output-path results/semantic_sentencebert_summary.csv

python evaluate_semantic_judge-all.py \
  --config configs/config_evaluate_semantic_judge.json
```

These scripts report BERTScore precision/recall/F1, MPNet cosine similarity, and optional Janus-based LLM judging. Small-sample variants are provided for environment checks before full evaluation.

### Token retrieval

```bash
python export_predicted_janus_tokens.py \
  --config configs/config_export_predicted_janus_tokens.json \
  --mode jepa

python evaluate_token_retrieval.py \
  --config configs/config_evaluate_token_retrieval.json \
  --save-details
```

The retrieval evaluation reports Top-k accuracy, mean rank, mean reciprocal rank (MRR), and positive/negative cosine similarity. Token export also supports Stage 1, diffusion, random-initialization, and random-noise controls.

## Ablation, robustness, and profiling

The provided shell scripts reproduce the main experiment families after their path variables are updated:

```bash
bash run_ablation_stage1.sh
bash run_ablation_stage2.sh
bash run_noise_robustness.sh
```

Profile parameter count, FLOPs, inference throughput, latency, and peak GPU memory with:

```bash
python profile_model.py --config configs/config_profile.json
```

## Reproducibility notes

- Models are trained independently for each subject–ROI pair.
- The reported NSD split contains 8,859 training stimuli and 982 test stimuli.
- Check that Stage-1 and Stage-2 configurations contain the same subject–ROI pairs.
- Keep generated texts and reference texts in exactly the same subject-specific sample order.
- On Linux, output paths are case-sensitive; for example, `results/ablation` and `results/Ablation` are different directories.
- The committed checkpoints, model weights, and source NSD arrays are not included in this code directory.
- Open-ended answers can contain unsupported details when the brain-derived representation is ambiguous; generated text should not be interpreted as uniquely attributable to the fMRI signal without appropriate controls.

## License

A license has not yet been added to this repository. Add the intended license file before public redistribution or reuse.
