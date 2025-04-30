from pathlib import PurePosixPath
from typing import Union

import modal

APP_NAME = "fast-edit"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

# Create an image with FastEdit dependencies
fastedit_image = (
    modal.Image.from_registry("python:3.10-slim")
    .apt_install("git")
    .pip_install(
        "torch>=1.13.1",
        "transformers>=4.29.1",
        "datasets>=2.12.0",
        "accelerate>=0.19.0",
        "huggingface_hub==0.23.2",
        "hf-transfer==0.1.5",
        "sentencepiece",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="false",
        )
    )
    .entrypoint([])
)

# Create a dedicated inference image with only what's needed for inference
inference_image = (
     modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install("vllm==0.5.1", "torch==2.3.0")
    .entrypoint([])
)

app = modal.App(
    APP_NAME,
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
    ],
)

# Volumes for pre-trained models and training runs
pretrained_volume = modal.Volume.from_name(
    "fast-edit-pretrained-vol", create_if_missing=True
)
runs_volume = modal.Volume.from_name(
    "fast-edit-runs-vol", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}

class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"