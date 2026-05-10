#!/usr/bin/env python3
"""Upload a local output directory to a HuggingFace model repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an output directory to a HuggingFace model repo."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Local directory to upload.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination model repo, for example: username/model-name.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Destination path inside the repo. Defaults to the repo root.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch or revision to upload to. Defaults to the repo default branch.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN or cached login.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    if not output_dir.is_dir():
        raise SystemExit(f"Output directory does not exist: {output_dir}")

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    commit_message = args.commit_message or f"Upload {output_dir.name}"
    commit_info = api.upload_folder(
        folder_path=str(output_dir),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=commit_message,
    )
    print(f"Uploaded {output_dir} to https://huggingface.co/{args.repo_id}")
    print(f"Commit: {commit_info.oid}")


if __name__ == "__main__":
    main()
