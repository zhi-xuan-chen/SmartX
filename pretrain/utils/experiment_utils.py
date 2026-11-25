from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys

import torch.distributed as dist
import yaml

from .utils import print_rank_zero

def setup_experiment(
    run_name, config, base_dir="experiments", resume_from_checkpoint=None
):
    """
    Set up a versioned experiment directory with complete code snapshot.
    In multi-GPU training, only rank 0 creates directories and saves files.

    Args:
        run_name (str): Name of the experiment run
        config (dict): Configuration dictionary
        base_dir (str): Base directory for experiments
        resume_from_checkpoint (str, optional): Path to checkpoint to resume from

    Returns:
        dict: Dictionary containing paths and experiment info
            - experiment_dir: Path to experiment directory
            - checkpoint_dir: Path to checkpoints
            - tb_dir: Path to tensorboard logs
            - code_dir: Path to code snapshot
            - version: Version number
            - wandb_run_id: W&B run ID (if resuming)
            - is_resume: Whether this is a resumed experiment
    """
    base_dir = Path(base_dir)

    # Check if we're in distributed training and not rank 0
    is_distributed = dist.is_initialized()
    is_rank_zero = not is_distributed or dist.get_rank() == 0

    # Handle resuming from checkpoint
    if resume_from_checkpoint:
        result = _handle_resume_from_checkpoint(resume_from_checkpoint)
        if result:
            return result

    # Create base and run directories
    run_dir = base_dir / run_name
    if is_rank_zero:
        base_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Determine version number and create experiment directory
        version = _get_next_version(run_dir)
        experiment_dir = run_dir / f"version_{version}"
        _create_experiment_structure(experiment_dir, config, run_name, version)
        print_rank_zero(f"Experiment directory: {experiment_dir}")

    if is_distributed:
        selected = [version if is_rank_zero else None]
        dist.broadcast_object_list(selected, src=0)
        version = selected[0]
        experiment_dir = run_dir / f"version_{version}"

    return {
        "experiment_dir": experiment_dir,
        "checkpoint_dir": experiment_dir / "checkpoints",
        "tb_dir": experiment_dir / "tensorboard",
        "code_dir": experiment_dir / "code_snapshot",
        "version": version,
        "wandb_run_id": None,
        "is_resume": False,
    }


def _handle_resume_from_checkpoint(resume_from_checkpoint):
    """Handle resuming from a checkpoint."""
    is_distributed = dist.is_initialized()
    is_rank_zero = not is_distributed or dist.get_rank() == 0
    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.exists():
        return None

    # Extract experiment directory from checkpoint path
    # Expected structure: .../run_name/version_X/checkpoints/checkpoint.ckpt
    experiment_dir = checkpoint_path.parent.parent
    if not (experiment_dir.exists() and experiment_dir.name.startswith("version_")):
        print_rank_zero(f"Warning: Invalid checkpoint path structure: {resume_from_checkpoint}")
        print_rank_zero("Expected structure: .../run_name/version_X/checkpoints/checkpoint.ckpt")
        return None

    version = int(experiment_dir.name.split("version_")[-1])

    # Load existing experiment metadata to get W&B run ID
    wandb_run_id = None
    metadata_file = experiment_dir / "code_snapshot" / "experiment_info.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                wandb_run_id = metadata.get("wandb_run_id")
        except Exception as e:
            print_rank_zero(f"Warning: Could not load experiment metadata: {e}")
    if is_rank_zero:
        print_rank_zero(f"Resuming experiment from: {experiment_dir}")
        print_rank_zero(f"W&B run ID: {wandb_run_id}")

    return {
        "experiment_dir": experiment_dir,
        "checkpoint_dir": experiment_dir / "checkpoints",
        "tb_dir": experiment_dir / "tensorboard",
        "code_dir": experiment_dir / "code_snapshot",
        "version": version,
        "wandb_run_id": wandb_run_id,
        "is_resume": True,
    }


def _get_next_version(run_dir):
    """Get the next version number for a new experiment."""
    existing_versions = [
        d.name
        for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("version_")
    ]
    if not existing_versions:
        return 0

    version_nums = [
        int(v.split("version_")[-1])
        for v in existing_versions
        if v.split("version_")[-1].isdigit()
    ]
    return max(version_nums) + 1 if version_nums else 0


def _get_latest_version(run_dir):
    """Get the latest version number for non-rank-0 processes."""
    if not run_dir.exists():
        return 0

    existing_versions = [
        d.name
        for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("version_")
    ]
    if not existing_versions:
        return 0

    version_nums = [
        int(v.split("version_")[-1])
        for v in existing_versions
        if v.split("version_")[-1].isdigit()
    ]
    return max(version_nums) if version_nums else 0


def _create_experiment_structure(experiment_dir, config, run_name, version, verbose=False):
    """Create the experiment directory structure and save metadata."""
    if verbose:
        print_rank_zero(f"[Rank 0] Creating experiment directory: {experiment_dir}")
    experiment_dir.mkdir(exist_ok=True)

    # Create subdirectories
    if verbose:
        print_rank_zero("[Rank 0] Creating subdirectories...")
    for subdir in ["tensorboard", "checkpoints", "code_snapshot"]:
        (experiment_dir / subdir).mkdir(exist_ok=True)

    code_dir = experiment_dir / "code_snapshot"

    # Save complete code snapshot
    if verbose:
        print_rank_zero("[Rank 0] Saving code snapshot (this may take time on NFS)...")
    _save_code_snapshot(code_dir)
    if verbose:
        print_rank_zero("[Rank 0] Code snapshot saved successfully")

    # Save experiment metadata
    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "run_name": run_name,
        "version": version,
        "command": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "python_version": sys.version,
    }

    with open(code_dir / "experiment_info.json", "w") as f:
        json.dump(experiment_info, f, indent=2)


def _save_code_snapshot(code_dir):
    """Save all Python files to the code snapshot directory."""
    source_root = Path(__file__).resolve().parents[2]
    destination = Path(code_dir).resolve()
    source_files = list(source_root.rglob("*.py"))
    for py_file in source_files:
        if destination == py_file or destination in py_file.parents:
            continue
        # Skip hidden directories and common ignore patterns
        if any(part.startswith(".") for part in py_file.parts):
            continue
        if any(
            part in ["__pycache__", "wandb", "tb_logs", "experiments", "outputs", "code_snapshot"]
            for part in py_file.parts
        ):
            continue

        relative_path = py_file.relative_to(source_root)
        target_file = code_dir / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_file, target_file)


def save_wandb_run_id(experiment_dir, wandb_run_id):
    """Save W&B run ID to experiment metadata for future resuming."""
    is_distributed = dist.is_initialized()
    is_rank_zero = not is_distributed or dist.get_rank() == 0
    if not is_rank_zero:
        return
    metadata_file = Path(experiment_dir) / "code_snapshot" / "experiment_info.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            metadata["wandb_run_id"] = wandb_run_id

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print_rank_zero(f"Saved W&B run ID {wandb_run_id} to experiment metadata")
        except Exception as e:
            print_rank_zero(f"Warning: Could not save W&B run ID: {e}")


_ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)


def _expand_config_env(value):
    if isinstance(value, str):

        def replace(match):
            name, default = match.groups()
            configured = os.environ.get(name)
            if configured:
                return configured
            if default is not None:
                return default
            raise ValueError(
                f"Environment variable {name} is required by the config"
            )

        return os.path.expanduser(_ENV_PATTERN.sub(replace, value))
    if isinstance(value, list):
        return [_expand_config_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_config_env(item) for key, item in value.items()}
    return value


def load_config(config_path):
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    config = _expand_config_env(config)
    # Ensure numeric values are properly typed
    numeric_keys = [
        "lr",
        "weight_decay",
        "mask_ratio",
        "dropout",
        "layer_decay",
        "warmup_steps",
        "total_steps",
        "head_lr",
        "base_lr",
        "vision_lr",
        "text_lr",
    ]
    for key in numeric_keys:
        if key in config and isinstance(config[key], str):
            try:
                config[key] = float(config[key])
            except ValueError:
                pass  # Keep as string if conversion fails

    return config
