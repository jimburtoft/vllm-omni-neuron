# SPDX-License-Identifier: Apache-2.0
"""Push or pull MiniMax-H3's compiled graphs to a Hugging Face repo, so a new host skips the compile.

    python examples/minimax_h3/neuron_cache.py pull <repo_id> [--subdir trn2-tp4-1344x768-124f]
    python examples/minimax_h3/neuron_cache.py push <repo_id> [--subdir ...]

The caches are `$VLLM_CACHE_ROOT/neuron/compile_cache` (the DiT, conditioner and decoder NEFFs) and
`$NKI_COMPILE_CACHE_URL` (the NKI kernels). Both are keyed on the traced graph and the compiler
arguments, so a pulled cache only hits for the same plugin commit, the same SDK image, the same
parallel layout and the same geometry (canvas, frame count, prompt-length bucket); a miss compiles
as usual. One `--subdir` per such combination.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download


def _dirs():
    vllm = os.environ.get("VLLM_CACHE_ROOT")
    nki = os.environ.get("NKI_COMPILE_CACHE_URL")
    if not vllm or not nki:
        sys.exit("Set VLLM_CACHE_ROOT and NKI_COMPILE_CACHE_URL first (the run uses the same values).")
    return {"compile_cache": os.path.join(vllm, "neuron", "compile_cache"), "nki": nki}


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
        for name, target in dirs.items():
            source = os.path.join(local, args.subdir, name)
            if not os.path.isdir(source):
                continue
            os.makedirs(target, exist_ok=True)
            os.system(f"cp -rn {source}/. {target}/")
            print(f"pulled {name} -> {target}")
        return
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    for name, source in dirs.items():
        if os.path.isdir(source):
            api.upload_folder(
                repo_id=args.repo_id, folder_path=source, path_in_repo=f"{args.subdir}/{name}",
                ignore_patterns=["*.lock", "locks/*"],
            )
            print(f"pushed {source} -> {args.repo_id}/{args.subdir}/{name}")


if __name__ == "__main__":
    main()
