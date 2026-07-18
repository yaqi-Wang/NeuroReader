import argparse
import json
import os

import h5py
import numpy as np
import scipy.io
from tqdm.auto import tqdm

from nsd_access import NSDAccess


def _collect_captions(nsda: NSDAccess, image_index: int, max_captions: int = 5):
    coco_infos = nsda.read_image_coco_info([image_index], info_type="captions")
    captions = [str(item["caption"]).strip() for item in coco_infos if str(item.get("caption", "")).strip()]
    captions = captions[:max_captions]
    while len(captions) < max_captions:
        captions.append("")
    return captions


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO caption references aligned to subject-specific NSD test order")
    parser.add_argument(
        "--imgidx",
        default=[0, 983],
        nargs=2,
        type=int,
        help="range in subject-internal test selection space, e.g. 0 982",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="subj01",
        help="subject folder name, e.g. subj01",
    )
    parser.add_argument(
        "--nsd-data-path",
        type=str,
        default="/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data",
        help="NSD root path",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default="/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main",
        help="project root that contains mrifeat_0526/{subject}/{subject}_stims_ave.npy",
    )
    parser.add_argument(
        "--stims-ave-path",
        type=str,
        default=None,
        help="optional explicit path to {subject}_stims_ave.npy",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default='/home/data/wangyaqi/projects/18BrainGPT/data/reference_texts/{subject}_caption_refs_0_981.npy',
        help="output .npy path",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        default='/home/data/wangyaqi/projects/18BrainGPT/data/reference_texts/{subject}_caption_refs_0_981.json',
        help="optional output .json path",
    )
    args = parser.parse_args()

    subject = args.subject
    nsd_data_path = args.nsd_data_path
    root_path = args.root_path

    stims_ave_path = args.stims_ave_path or f"{root_path}/mrifeat_0526/{subject}/{subject}_stims_ave.npy"
    output_path = args.output_path
    output_json_path = args.output_json_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)

    nsd_expdesign = scipy.io.loadmat(f"{nsd_data_path}/nsddata/experiments/nsd/nsd_expdesign.mat")
    sharedix = nsd_expdesign["sharedix"] - 1

    stims_ave = np.load(stims_ave_path)

    tr_idx = np.zeros_like(stims_ave)
    shared_set = set(sharedix.flatten().tolist())
    for idx, stim_id in enumerate(stims_ave.tolist()):
        tr_idx[idx] = 0 if stim_id in shared_set else 1

    test_indices = np.where(tr_idx == 0)[0]

    sel_start, sel_end = args.imgidx
    sel_end = min(sel_end, len(test_indices))
    if sel_start < 0 or sel_start >= sel_end:
        raise ValueError(f"Invalid imgidx range: {args.imgidx}, len(test_indices)={len(test_indices)}")

    nsda = NSDAccess(nsd_data_path)
    sf = h5py.File(nsda.stimuli_file, "r")
    sdataset = sf.get("imgBrick")
    if sdataset is None:
        raise ValueError(f"Unable to locate imgBrick in {nsda.stimuli_file}")

    caption_refs = []
    resolved_indices = []

    for current_selection_idx in tqdm(range(sel_start, sel_end), desc=f"Loading captions for {subject}"):
        single_imgidx_te = int(test_indices[current_selection_idx])
        single_idx73k = int(stims_ave[single_imgidx_te])
        _ = sdataset[single_idx73k, :, :, :]
        captions = _collect_captions(nsda, single_idx73k, max_captions=5)
        caption_refs.append(captions)
        resolved_indices.append(
            {
                "selection_index": int(current_selection_idx),
                "subject_internal_index": single_imgidx_te,
                "nsd_image_index": single_idx73k,
            }
        )

    caption_array = np.asarray(caption_refs, dtype=object)
    np.save(output_path, caption_array)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "subject": subject,
                    "selection_range": [sel_start, sel_end],
                    "num_samples": len(caption_refs),
                    "resolved_indices": resolved_indices,
                    "captions": caption_refs,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    sf.close()
    print(f"Saved caption references to {output_path} with shape {caption_array.shape}")


if __name__ == "__main__":
    main()
