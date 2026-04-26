#!/usr/bin/env python
"""
Merge one or more local LeRobot datasets and push the result to HuggingFace.

Usage:
    python agentic-xle/push_dataset.py \\
        --datasets rlodhi/screw_place rlodhi/screw_place_p2 \\
        --repo-id rlodhi/screw_placing \\
        [--merged-dir ~/my_merged] \\
        [--tags robotics lerobot] \\
        [--private] \\
        [--no-upload-large-folder]

Before running:
    huggingface-cli login   # or: export HF_TOKEN=hf_...
    conda activate lerobot
"""

import argparse
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def parse_args():
    p = argparse.ArgumentParser(description="Merge local LeRobot datasets and push to HF.")
    p.add_argument(
        "--datasets", nargs="+", required=True, metavar="REPO_ID",
        help="One or more local dataset repo-ids (e.g. rlodhi/screw_place rlodhi/screw_place_p2)",
    )
    p.add_argument(
        "--repo-id", required=True,
        help="HuggingFace repo-id to push the merged dataset to (e.g. rlodhi/screw_placing)",
    )
    p.add_argument(
        "--merged-dir", type=Path, default=None,
        help="Where to write the merged dataset locally (default: ~/.<first-dataset-name>_merged)",
    )
    p.add_argument(
        "--tags", nargs="*", default=["robotics", "lerobot"],
        help="Tags for the HF dataset card (default: robotics lerobot)",
    )
    p.add_argument(
        "--private", action="store_true",
        help="Push as a private repository (default: public)",
    )
    p.add_argument(
        "--no-upload-large-folder", action="store_true",
        help="Disable chunked upload (not recommended for large video datasets)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load each dataset from the local cache
    datasets = []
    for repo_id in args.datasets:
        root = HF_LEROBOT_HOME / repo_id
        print(f"Loading {repo_id} from {root} …")
        datasets.append(LeRobotDataset(repo_id=repo_id, root=root))

    if len(datasets) == 1:
        # Nothing to merge — push directly
        merged = datasets[0]
        print("Only one dataset provided, skipping merge step.")
    else:
        # Derive a sensible default output dir from the target repo name
        merged_dir = args.merged_dir or (
            Path.home() / (args.repo_id.replace("/", "_") + "_merged")
        )
        print(f"Merging {len(datasets)} datasets into {merged_dir} …")
        merged = merge_datasets(
            datasets=datasets,
            output_repo_id=args.repo_id,
            output_dir=merged_dir,
        )

    print(f"Dataset ready: {merged.num_episodes} episodes, {merged.num_frames} frames")

    # Push to HuggingFace
    print(f"Pushing to HuggingFace as {args.repo_id} …")
    merged.push_to_hub(
        tags=args.tags,
        private=args.private,
        upload_large_folder=not args.no_upload_large_folder,
    )

    print(f"\nDone! Dataset live at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
