"""Engine-level PersonaPlex check: MoshiStage0.prefill_prompt + free-running greedy steps on the clip, vs the NVIDIA
reference's greedy run with the same prompt (pp goldens). Reports agreement of the emitted text / agent tokens per
frame (free-running: diverges once the first token differs) and the first divergence frame, and the text."""
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device
    from moshi_neuron.engine import MoshiStage0
    neuron_runtime_env()
    dev = torch_device()
    g = torch.load(sys.argv[1])
    wav = torch.from_numpy(np.load(os.path.expanduser("~/goldens/clip_utt2.npy")).astype(np.float32))
    B = int(os.environ.get("B", "2"))
    s0 = MoshiStage0(B=B, device=dev, compile_fn=compile_fn(), greedy=True, seed=0)
    s0.reset(range(B))
    act = torch.zeros(B, dtype=torch.bool); act[0] = True
    if os.environ.get("PREFILL", "0") == "1":
        P = s0.prefill_batched(0, g["voice"], g["prompt"])
    elif os.environ.get("QUEUED", "0") == "1":
        P = s0.start_prompt(0, g["voice"], g["prompt"])
        while s0.prompting(0):
            s0.step(torch.zeros(B, 1920), act)
    else:
        P = s0.prefill_prompt(0, g["voice"], g["prompt"])
    print("prompt steps", P, "prefill" if os.environ.get("PREFILL") == "1" else ("queued" if os.environ.get("QUEUED") == "1" else "inline"), flush=True)
    T = g["tokens"].shape[0]
    ours = []
    for t in range(T):
        pcm = torch.zeros(B, 1920); pcm[0] = wav[t * 1920:(t + 1) * 1920]
        raw, out, ready, _ = s0.step(pcm, act)
        ours.append(raw[0].clone())
    ours = torch.stack(ours)                       # [T, 9] raw (text + 8 agent) per step
    ref = g["tokens"]                              # [T, 17] reference output (de-delayed) per step
    # compare the model's sampled text (raw col 0) to the reference's text input of the next step (= its sampled text)
    rt = g["inputs"][1:, 0]; ot = ours[:-1, 0]
    agree = (rt == ot).float()
    first = int((agree == 0).nonzero()[0]) if (agree == 0).any() else T
    print(f"text agreement {agree.mean():.3f}, first divergence at frame {first}")
    ra = g["inputs"][1:, 1:9]; oa = ours[:-1, 1:9]
    print(f"agent cb0 agreement {(ra[:, 0] == oa[:, 0]).float().mean():.3f}, all-8 {(ra == oa).float().mean():.3f}")
    from huggingface_hub import hf_hub_download
    import sentencepiece
    tok = sentencepiece.SentencePieceProcessor(hf_hub_download("nvidia/personaplex-7b-v1", "tokenizer_spm_32k_3.model"))
    txt = lambda ids: "".join(tok.id_to_piece(int(i)).replace("\u2581", " ") for i in ids if int(i) not in (0, 3, -2, 32000))
    print("ours:", txt(ours[:, 0])[:300]); print("ref :", txt(g["inputs"][:, 0])[:300])
