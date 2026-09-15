"""Utilities: seeds, run manifests, environment snapshots."""

from .seeds import derive_seed, set_all_seeds, seed_record_for_config
from .manifest import (
    build_manifest,
    experiment_folder,
    finalize_manifest,
    find_repo_root,
    freeze_config,
    git_info,
    local_now_iso,
    make_experiment_dir,
    new_run_id,
    platform_tag_from_gpu_name,
    read_json,
    sha256_file,
    sha256_text,
    slugify,
    utc_now_iso,
    write_json,
)
from .envsnap import snapshot_environment

__all__ = [
    "derive_seed", "set_all_seeds", "seed_record_for_config",
    "build_manifest", "experiment_folder", "finalize_manifest", "find_repo_root", "freeze_config",
    "git_info", "local_now_iso", "make_experiment_dir", "new_run_id",
    "platform_tag_from_gpu_name", "read_json", "sha256_file", "sha256_text",
    "slugify", "utc_now_iso", "write_json", "snapshot_environment",
]
