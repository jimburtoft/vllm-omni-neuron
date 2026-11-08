"""PersonaPlex golden capture from the NVIDIA personaplex reference (CPU), with system prompt (voice + text).

Runs offline.py's protocol: voice prompt (.pt embeddings), 0.5 s silence, text prompt "<system> ... <system>", 0.5 s
silence (stepped frame by frame: no prefill, no pinning, so the KV is the plain sliding window), then the user clip
frame by frame. Records, for every LM step after the prompt, the transformer input tokens [17] (what the model
consumed), text logits, depformer logits for the 8 agent steps, output tokens; plus the prompt length in frames.

Our engine replays: prompt phase from the recorded per-frame input tokens (all forced), then user frames
teacher-forced on the reference's sampled tokens, comparing argmax agreement.

usage (CPU box): python make_pp_goldens.py --out pp_goldens --voice NATF2 --prompt "You are a wise and friendly teacher..."
"""
import argparse
import os
import time

import numpy as np
import torch

SR = 24000
FRAME = 1920


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pp_goldens")
    ap.add_argument("--voice", default="NATF2")
    ap.add_argument("--prompt", default="You are a wise and friendly teacher. Answer questions or provide advice in a "
                                        "clear and engaging way.")
    ap.add_argument("--clips", default="utt2")
    ap.add_argument("--goldens", default=os.path.expanduser("~/goldens"))
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    from huggingface_hub import hf_hub_download
    import sentencepiece
    import tarfile
    from moshi.models import loaders
    from moshi.models.lm import LMGen
    from moshi.offline import wrap_with_system_tags

    repo = "nvidia/personaplex-7b-v1"
    mimi = loaders.get_mimi(hf_hub_download(repo, loaders.MIMI_NAME), "cpu")
    other = loaders.get_mimi(hf_hub_download(repo, loaders.MIMI_NAME), "cpu")
    tok = sentencepiece.SentencePieceProcessor(hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME))
    lm = loaders.get_moshi_lm(hf_hub_download(repo, loaders.MOSHI_NAME), device="cpu")
    lm = lm.to(getattr(torch, args.dtype)).eval()
    vdir = os.path.join(args.out, "voices")
    if not os.path.isdir(vdir):
        with tarfile.open(hf_hub_download(repo, "voices.tgz")) as t:
            t.extractall(args.out)
    os.makedirs(args.out, exist_ok=True)

    rec = {"inputs": [], "emb": [], "text_logits": [], "dep_logits": [], "tokens": []}
    cur_dep = []
    orig_fd = lm.forward_depformer

    def fd(cb, seq, tout):
        lg = orig_fd(cb, seq, tout)
        cur_dep.append(lg[0, 0, 0].float().clone())
        return lg
    lm.forward_depformer = fd
    orig_emb = lm.forward_embeddings

    def fe(x):
        # every transformer step (prompt and user frames) goes through forward_embeddings: record the input embedding
        rec["emb"].append(x[0, -1].float().clone())
        out = orig_emb(x)
        rec["_tl"] = out[1][0, 0, 0].float().clone()
        return out
    lm.forward_embeddings = fe
    orig_codes = lm.forward_codes

    def fc(seq):
        rec["inputs"].append(seq[0, :, -1].clone())     # transformer input tokens this step [17]
        return orig_codes(seq)
    lm.forward_codes = fc

    gen = LMGen(lm, audio_silence_frame_cnt=int(0.5 * mimi.frame_rate), sample_rate=mimi.sample_rate, device="cpu",
                frame_rate=mimi.frame_rate, use_sampling=False, prefill_system_prompt=False, pin_system_prompt=False)
    gen.streaming_forever(1)
    mimi.streaming_forever(1)
    other.streaming_forever(1)
    gen.load_voice_prompt_embeddings(os.path.join(vdir, f"{args.voice}.pt"))
    gen.voice_prompt_embeddings = gen.voice_prompt_embeddings.to(getattr(torch, args.dtype))
    gen.text_prompt_tokens = tok.encode(wrap_with_system_tags(args.prompt))
    mimi.reset_streaming(); other.reset_streaming(); gen.reset_streaming()
    t0 = time.time()
    gen.step_system_prompts(mimi)
    n_prompt = len(rec["emb"])
    print(f"prompt: {n_prompt} transformer steps ({len(rec['inputs'])} from tokens), text prompt "
          f"{len(gen.text_prompt_tokens)} tokens, {time.time() - t0:.0f}s", flush=True)
    prompt_emb = torch.stack(rec["emb"])                       # [P, 4096] fp32: what the transformer consumed
    prompt_cache = gen._streaming_state.cache.clone()          # token ring after the prompt (delays)
    prompt_offset = int(gen._streaming_state.offset)
    mimi.reset_streaming()
    for name in args.clips.split(","):
        wav = np.load(os.path.join(args.goldens, f"clip_{name}.npy")).astype(np.float32)
        x = torch.from_numpy(wav)[None, None]
        rec.update(inputs=[], emb=[], text_logits=[], dep_logits=[], tokens=[], user_codes=[])
        for i, t in enumerate(range(0, x.shape[-1], FRAME)):
            codes = mimi.encode(x[..., t:t + FRAME])
            rec["user_codes"].append(codes[0, :, 0].clone())
            cur_dep.clear()
            out = gen.step(codes[:, :, :1])
            rec["dep_logits"].append(torch.stack(cur_dep[:8]) if len(cur_dep) >= 8 else torch.zeros(8, 2048))
            rec["text_logits"].append(rec.get("_tl") if rec.get("_tl") is not None else torch.zeros(32000))
            rec["tokens"].append(out[0, :, 0].clone() if out is not None else torch.full((17,), -2, dtype=torch.long))
            if i % 25 == 0:
                print(f"  frame {i} {(time.time() - t0) / (i + 1):.2f}s/frame", flush=True)
        res = {k: torch.stack(v) for k, v in rec.items() if k in ("inputs", "text_logits", "dep_logits", "tokens",
                                                                  "user_codes")}
        res["prompt_emb"], res["prompt_cache"], res["prompt_offset"] = prompt_emb, prompt_cache, prompt_offset
        res["voice"], res["prompt"] = args.voice, args.prompt
        torch.save(res, os.path.join(args.out, f"pp_{name}_{args.dtype}_greedy.pt"))
        text = "".join(tok.id_to_piece(int(t)).replace("\u2581", " ") for t in res["tokens"][:, 0]
                       if int(t) not in (0, 3, -2))
        print(name, "inputs", tuple(res["inputs"].shape), "prompt", tuple(prompt_emb.shape), "offset", prompt_offset,
              "text:", text[:300],
              flush=True)


if __name__ == "__main__":
    main()
