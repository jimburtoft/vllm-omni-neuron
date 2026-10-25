# SPDX-License-Identifier: Apache-2.0
"""Host weight copies backed by a file on local disk instead of anonymous process memory.

A single trn2 device keeps ~75 GB of host-side weight copies for MiniMax-H3: the streamed conditioner
shards, the DiT blocks evicted around the decode, and the video decoder that is swapped in. In process
memory those pages are anonymous: under memory pressure the kernel can only write them to swap and read
them back slowly. `MINIMAX_H3_HOST_WEIGHTS_DIR=<dir>` (a local NVMe instance store, e.g. `/mnt/nvme`)
writes each copy to a file there and memory-maps it read-only instead:

* The pages are page cache, not process memory: the kernel drops them under pressure without writing
  anything, and re-reads them from the file on the next use.
* Ranks whose copies are identical (the video decoder is replicated on every rank) map one file, so the
  page cache holds one copy for all of them. Content-addressed by a caller-provided key.
* On an idle host nothing changes: the pages stay cached and reads are memory speed.

The copy to the device reads straight from the mapped pages (`tensor.to(device)`); no extra host buffer
is kept.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

DIR_ENV = "MINIMAX_H3_HOST_WEIGHTS_DIR"
_ALIGN = 4096


def host_weights_dir() -> str | None:
    value = os.environ.get(DIR_ENV, "").strip()
    return value or None


def to_mapped(tensors: dict, key: str, directory: str) -> dict:
    """Return read-only CPU views of ``tensors`` backed by ``<directory>/<key>.bin``.

    ``tensors`` maps any hashable to a CPU tensor. The first process to arrive writes the file (others
    with the same ``key`` wait on a lock and reuse it). The returned tensors share memory with the file
    mapping; the inputs can be released.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{key}.bin")
    names = list(tensors)
    layout, offset = [], 0
    for name in names:
        t = tensors[name]
        nbytes = t.numel() * t.element_size()
        layout.append((offset, nbytes, t.dtype, tuple(t.shape)))
        offset = (offset + nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    total = max(offset, _ALIGN)
    meta = json.dumps([(o, n, str(d), s) for o, n, d, s in layout])
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = path + ".done"
        if not (os.path.exists(done) and open(done).read() == meta):
            with open(path + ".tmp", "wb") as handle:
                handle.truncate(total)
                for name, (o, n, _, _) in zip(names, layout):
                    handle.seek(o)
                    handle.write(tensors[name].contiguous().view(torch.uint8).reshape(-1).numpy().tobytes())
            os.replace(path + ".tmp", path)
            with open(done, "w") as handle:
                handle.write(meta)
        fcntl.flock(lock, fcntl.LOCK_UN)
    storage = torch.UntypedStorage.from_file(path, shared=False, nbytes=total)
    flat = torch.empty(0, dtype=torch.uint8).set_(storage)
    out = {}
    for name, (o, n, dtype, shape) in zip(names, layout):
        out[name] = flat[o : o + n].view(dtype).view(shape)
    logger.info("MiniMax-H3 host weights: %s, %.2f GB mapped from %s.", key, total / 1e9, path)
    return out
