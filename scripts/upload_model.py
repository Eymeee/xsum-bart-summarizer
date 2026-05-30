"""Upload the fine-tuned BART checkpoint and model card to Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi


REPO_ID = "Eymeee/xsum-bart-summarizer"
MODEL_DIR = Path("models/bart_xsum_finetuned")
MODEL_CARD = Path("deploy/model/README.md")


def main() -> None:
    api = HfApi()
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(MODEL_DIR),
        path_in_repo=".",
        commit_message="Upload fine-tuned BART checkpoint",
    )
    api.upload_file(
        repo_id=REPO_ID,
        repo_type="model",
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        commit_message="Add model card",
    )
    print(f"Uploaded model artifacts to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
