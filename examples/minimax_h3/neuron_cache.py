# SPDX-License-Identifier: Apache-2.0
"""Push or pull MiniMax-H3's compiled graphs to a Hugging Face repo, so a new host skips the compile.

    python examples/minimax_h3/neuron_cache.py pull <repo_id> [--subdir trn2-tp4-1344x768-124f]
    python examples/minimax_h3/neuron_cache.py push <repo_id> [--subdir ...]

The caches are `$TORCH_NEURONX_NEFF_CACHE_DIR` (default `/tmp/neff_cache`: the DiT, conditioner and
decoder NEFFs) and `$NKI_COMPILE_CACHE_URL` (the NKI kernels). Both are keyed on the traced graph and the compiler
arguments, so a pulled cache only hits for the same plugin commit, the same SDK image, the same
parallel layout and the same geometry (canvas, frame count, prompt-length bucket); a miss compiles
as usual. One `--subdir` per such combination.
"""

import argparse
import os
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download


def _dirs():
    # The native backend (`neuron_native_lite`, what this pipeline compiles with) writes its NEFFs to
    # TORCH_NEURONX_NEFF_CACHE_DIR, default /tmp/neff_cache; VLLM_CACHE_ROOT holds only the
    # neuron_libtorch backend's cache, which this pipeline does not use.
    nki = os.environ.get("NKI_COMPILE_CACHE_URL")
    if not nki:
        sys.exit("Set NKI_COMPILE_CACHE_URL first (the run uses the same value).")
    dirs = {"neff": os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache"), "nki": nki}
    vllm = os.environ.get("VLLM_CACHE_ROOT")
    if vllm:
        dirs["compile_cache"] = os.path.join(vllm, "neuron", "compile_cache")
    return dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("push", "pull"))
    parser.add_argument("repo_id")
    parser.add_argument("--subdir", default="trn2-tp4-1344x768-124f")
    parser.add_argument("--private", action="store_true", help="push: create the repo private.")
    args = parser.parse_args()
    dirs = _dirs()
    if args.action == "pull":
        local = snapshot_download(args.repo_id, allow_patterns=[f"{args.subdir}/*"], max_workers=16)
        if not os.path.isdir(os.path.join(local, args.subdir)):
            sys.exit(f"{args.repo_id} has no {args.subdir}/.")
        for name, target in dirs.items():
            source = os.path.join(local, args.subdir, name)
            if not os.path.isdir(source):
                continue
            # The hub snapshot is relative symlinks into its blob store: copy the files, not the links.
            shutil.copytree(source, target, symlinks=False, dirs_exist_ok=True)
            print(f"pulled {name} -> {target}")
        return
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    for name, source in dirs.items():
        if os.path.isdir(source):
            api.upload_folder(
                repo_id=args.repo_id, folder_path=source, path_in_repo=f"{args.subdir}/{name}",
                ignore_patterns=["*.lock", "locks/*", "*.tmp"],
            )
            print(f"pushed {source} -> {args.repo_id}/{args.subdir}/{name}")


if __name__ == "__main__":
    main()
