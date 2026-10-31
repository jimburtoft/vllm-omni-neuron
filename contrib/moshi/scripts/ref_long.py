"""kyutai reference (CPU) on a long wav, server.py protocol; writes output wav + per-10s RMS + text."""
import sys, time
import numpy as np, soundfile as sf, torch
from moshi.models import LMGen, loaders
wav_in, out, dtype, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
torch.manual_seed(4242)
info = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
mimi = info.get_mimi(device="cpu"); lm = info.get_moshi(device="cpu", dtype=getattr(torch, dtype))
gen = LMGen(lm, use_sampling=(mode == "sampled"))
tok = info.get_text_tokenizer()
x, sr = sf.read(wav_in, dtype="float32"); x = torch.from_numpy(x)[None, None]
pcm, text = [], []
t0 = time.time()
with torch.no_grad(), mimi.streaming(1), gen.streaming(1):
    for i, t in enumerate(range(0, x.shape[-1] - 1919, 1920)):
        c = mimi.encode(x[..., t:t + 1920])
        if i == 0:
            mimi.reset_streaming()
        o = gen.step(c[:, :, :1])
        if o is None:
            continue
        pcm.append(mimi.decode(o[:, 1:])[0, 0].numpy())
        tt = int(o[0, 0, 0])
        if tt not in (0, 3):
            text.append(tok.id_to_piece(tt).replace("\u2581", " "))
        if i % 250 == 0:
            print(i, f"{(time.time() - t0) / (i + 1):.2f}s/frame", flush=True)
y = np.concatenate(pcm); sf.write(out + ".wav", y, 24000)
open(out + ".txt", "w").write("".join(text))
print("rms", " ".join(f"{float(np.sqrt((y[i*240000:(i+1)*240000]**2).mean())):.3f}" for i in range(len(y) // 240000)))
print("text", "".join(text)[:2000])
