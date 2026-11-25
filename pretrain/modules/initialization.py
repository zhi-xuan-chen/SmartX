"""Initialize model parameters from a released SmartX checkpoint."""
from pathlib import Path
import torch


def initialize_from_smartx(model, checkpoint="auto"):
    if checkpoint in ("auto", ""):
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(repo_id="Trusure/SmartX", filename="smartx_large.ckpt")
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SmartX checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    state = {key.removeprefix("module.").removeprefix("model."): value
             for key, value in state.items()}
    # Strict loading rejects missing, unexpected and shape-mismatched parameters.
    model.load_state_dict(state, strict=True)
    model.train()
    return str(path)
