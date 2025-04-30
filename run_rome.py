import os
import json

from typing import Optional, List, Dict

from .rome import ROMEHyperParams, apply_rome_to_model
from .utils.prints import print_loud
from .utils.template import Template
from .utils.mtloader import load_model_and_tokenizer
from .utils.generate import generate_fast

from .common import (
    app, 
    fastedit_image, 
    VOLUME_CONFIG,
    HOURS
)

# GPU config
GPU_CONFIG = "L40S:1"  # Using 1 L40S GPU

def ensure_model_cached(model_name: str):
    """Check if model is already in pretrained volume, download if not."""
    from huggingface_hub import snapshot_download
    
    try:
        # Try to use cached model
        snapshot_download(model_name, local_files_only=True, cache_dir="/pretrained")
        print(f"Using cached model {model_name} from pretrained volume.")
    except Exception:
        # Download model and save to volume
        print(f"Downloading {model_name} to pretrained volume...")
        snapshot_download(model_name, cache_dir="/pretrained")
        print(f"Model {model_name} downloaded and cached.")
        
        # Commit changes to volume
        VOLUME_CONFIG["/pretrained"].commit()

@app.function(
    image=fastedit_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def run_rome(
    requests: List[Dict[str, str]], model: str, config: str, template: Optional[str] = "default", checkpointing: Optional[bool] = False
) -> None:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        requests (`List[Dict[str, str]]`):
            The samples for editing.
        model (`str`):
            The name or path of the pre-trained transformer model to be edited.
        config (`str`):
            The name of the hyper-parameters to use for editing the model.
        template (`str`, *optional*, defaults to `default`):
            The name of the template to use in generation.
        output (`str`, *optional*, defaults to `None`):
            The path to save the edited model.
        checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing or not.
    """
    ensure_model_cached(model)

    queries = [query for request in requests for query in request["queries"]]

    model_old, tokenizer, batch_first = load_model_and_tokenizer(model, checkpointing)
    template = Template(name=template)

    print_loud("Retrieving hyperparameters")
    hparams = ROMEHyperParams.from_name(config)
    print(hparams)

    if len(queries) > 0:
        print_loud("Generating pre-update text")
        pre_update_text = generate_fast(model_old, tokenizer, queries, template, max_length=100)
        print("\n\n".join([queries[i] + " " + pre_update_text[i] for i in range(len(queries))]))

    print_loud(f"Applying rome to model")
    model_new, _ = apply_rome_to_model(
        model_old,
        tokenizer,
        requests,
        hparams,
        batch_first,
        return_diff_weights=False
    )

    if len(queries) > 0:
        print_loud("Generating post-update text")
        post_update_text = generate_fast(model_new, tokenizer, queries, template, max_length=100)
        print("\n\n".join([queries[i] + " " + post_update_text[i] for i in range(len(queries))]))


    print_loud("Saving model")
    model_new.config.use_cache = True
    model_new.save_pretrained("/runs/rome", max_shard_size="10GB")
    tokenizer.save_pretrained("/runs/rome")
    VOLUME_CONFIG["/runs"].commit()
    print(f"Model saved to /runs/rome")


@app.local_entrypoint()
def launch(
    data: str = os.path.join(os.path.dirname(__file__), "data", "example.json"), 
    model: str = "EleutherAI/gpt-j-6B", 
    config: str = "gpt-j-6b", 
    template: Optional[str] = "default",
    checkpointing: Optional[bool] = False
):
    assert os.path.exists(data), "data not found"
    with open(data, "r", encoding="utf-8") as f:
        requests = json.load(f)
    run_rome.remote(requests, model, config, template, checkpointing)

