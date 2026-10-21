"""fp32 CPU equivalence of our modules vs the official `qwen_tts` package (skipped if it is not installed).

Checks prompt construction, talker prefill/decode, code-predictor frame codes, fused engine == unbatched engine,
and Code2Wav vs the official decoder.
"""
import pytest
import torch

from conftest import CASES

qwen_tts = pytest.importorskip("qwen_tts")


@pytest.fixture(scope="module")
def ref(model_dir):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return qwen_tts.Qwen3TTSModel.from_pretrained(model_dir, device_map="cpu", dtype=torch.float32,
                                                  attn_implementation="eager")


@pytest.fixture(scope="module")
def ours(model_dir):
    from qwen3_tts_neuron.pipeline import Qwen3TTSNeuron
    return Qwen3TTSNeuron(model_dir, device="cpu", batch_size=2, prefill_buckets=(64, 128))


def _hf_prompt(ref, c):
    cap = {}
    talker = ref.model.talker
    orig = talker.generate

    def grab(*a, **kw):
        cap.update(emb=kw["inputs_embeds"], pad=kw["tts_pad_embed"])
        raise StopIteration
    talker.generate = grab
    try:
        ref.generate_voice_design(text=c["text"], language=c["language"], instruct=c["instruct"], do_sample=False,
                                  subtalker_dosample=False, max_new_tokens=4)
    except StopIteration:
        pass
    finally:
        talker.generate = orig
    return cap["emb"], cap["pad"]


@pytest.mark.parametrize("case", CASES, ids=[c["tag"] for c in CASES])
def test_prompt(ref, ours, case):
    emb, pad = _hf_prompt(ref, case)
    mine, mpad = ours.prompt(case["text"], case["instruct"], case["language"])
    assert mine.shape == emb.shape
    assert (mine - emb).abs().max() < 1e-4
    assert (mpad - pad).abs().max() < 1e-4


def test_talker_and_code_predictor(ref, ours):
    emb, _ = _hf_prompt(ref, CASES[0])
    talker = ref.model.talker
    T = emb.shape[1]
    with torch.no_grad():
        out = talker.model(inputs_embeds=emb, attention_mask=torch.ones(1, T, dtype=torch.long), use_cache=True)
        hf_h = out.last_hidden_state[:, -1:]
        hf_logits = talker.codec_head(hf_h).float().view(1, -1)
        ours.eng.talker.reset()
        h, lg = ours.eng.talker.step(emb, torch.arange(T))
        assert (h - hf_h).abs().max() < 1e-3 and (lg - hf_logits).abs().max() < 1e-3
        cb0 = torch.tensor([[int(hf_logits.argmax())]])
        e0 = talker.get_input_embeddings()(cb0)
        r = talker.code_predictor.generate(inputs_embeds=torch.cat((hf_h, e0), 1), max_new_tokens=15, do_sample=False,
                                           output_hidden_states=True, return_dict_in_generate=True)
        codes, _ = ours.eng.cp.frame(hf_h, e0)
    assert torch.equal(r.sequences.view(-1), codes)


def test_fused_matches_unbatched(ours):
    reqs = [ours.prompt(c["text"], c["instruct"], c["language"]) for c in CASES]
    base = [ours.eng.generate(e, p, max_new=30) for e, p in reqs]
    out, _, _ = ours.fe.run(reqs, max_new=30)  # B=2 slots, 3 requests -> exercises refill
    for o, b in zip(out, base):
        assert torch.equal(o["codes"], b)


def test_code2wav(ref, ours):
    codes = ours.generate_codes([CASES[0]["text"]], [CASES[0]["instruct"]], [CASES[0]["language"]], max_new=40)[0]
    with torch.no_grad():
        hf = ref.model.speech_tokenizer.model.decoder.chunked_decode(codes.t().unsqueeze(0), chunk_size=25,
                                                                     left_context_size=25).view(-1).float()
    mine = ours.c2w.decode(codes)
    snr = 10 * torch.log10((hf ** 2).sum() / ((mine - hf) ** 2).sum().clamp_min(1e-20))
    assert snr > 60, float(snr)


def test_rows_frame_greedy_matches_fused(ours):
    """FusedFrameRows with all-greedy rows must reproduce FusedFrame (and therefore the unbatched engine) exactly."""
    from qwen3_tts_neuron.fused import FusedFrameRows
    fe = ours.fe
    rows = FusedFrameRows(ours.eng, fe.bt, vec_cp=True)
    B, V, H = fe.B, ours.eng.talker.cfg.vocab, ours.eng.talker.cfg.hidden
    g = torch.Generator().manual_seed(0)
    h = torch.randn(B, 1, H, generator=g) * 0.1
    lg = torch.randn(B, V, generator=g)
    seen = torch.zeros(B, V, dtype=torch.bool); seen[:, ::7] = True
    allow = torch.ones(B, dtype=torch.bool); pos = torch.full((B,), 40)
    z = torch.zeros(B); zk = torch.zeros(B, dtype=torch.long)
    noise0 = torch.randn(B, V); noises = torch.randn(B, 15, 1, ours.eng.cp.cfg.vocab)
    for b in range(B):  # identical KV state for both runs
        pass
    snap = [(getattr(fe.bt, f"ck{i}").clone(), getattr(fe.bt, f"cv{i}").clone()) for i in range(28)]
    a = fe.ff(h, lg, seen, allow, h, pos)
    for i, (k, v) in enumerate(snap):
        getattr(fe.bt, f"ck{i}").copy_(k); getattr(fe.bt, f"cv{i}").copy_(v)
    r = rows(h, lg, seen, allow, h, pos, noise0, noises, torch.full((B,), 1.05), z, zk, z, zk)
    assert torch.equal(a[3], r[3])
    assert torch.allclose(a[1], r[1])


def test_chunked_prefill_and_ctx_buckets(model_dir):
    """Prompt split over small prefill buckets + decode with a context bucket == single-shot prefill + full context."""
    from qwen3_tts_neuron.batched import BatchedTalker, prefill_plan
    from qwen3_tts_neuron.engine import TTSEngine
    eng = TTSEngine(model_dir, device="cpu", max_ctx=256, dtype=torch.float32, compile=False, prefill_buckets=(64,))
    bt = BatchedTalker(eng.talker, 2, 256)
    g = torch.Generator().manual_seed(0)
    T = 150
    emb = torch.randn(1, T, 2048, generator=g) * 0.05

    def run(buckets, ctx):
        for i in range(28):
            getattr(bt, f"ck{i}").zero_(); getattr(bt, f"cv{i}").zero_()
        b = torch.tensor([1])
        for start, n, Tb, p in prefill_plan(T, buckets, 256):
            x = torch.zeros(1, Tb, 2048); x[:, Tb - n:] = emb[:, start:start + n]
            hl, lg = bt.prefill(x, b, p)
        x1 = torch.randn(2, 1, 2048, generator=torch.Generator().manual_seed(1)) * 0.05
        h2, lg2 = bt.decode(x1, torch.tensor([0, T]), ctx=ctx)
        return lg, lg2[1]

    a = run((256,), None)        # one shot (pads into the top slots), full context
    c = run((32, 64), 160)       # 64+64+(22 padded to 32), context bucket 160
    assert (a[0] - c[0]).abs().max() < 1e-4
    assert (a[1] - c[1]).abs().max() < 1e-4
