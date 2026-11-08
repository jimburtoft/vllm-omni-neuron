"""PersonaPlex system prompt (voice + text persona), reproducing NVIDIA personaplex `LMGen.step_system_prompts`
(prefill_system_prompt semantics: every stream forced, text/depformer outputs discarded).

Sequence per session (offline.py / server.py): voice-prompt frames (stored input embeddings from voices/<V>.pt,
replayed with step_embeddings), 0.5 s silence (6 frames: agent audio = SILENCE codes, user = SINE codes, text = 3),
text prompt tokens of "<system> {persona} <system>" (agent = SILENCE, user = SINE), 0.5 s silence. After that the
token ring (`state.cache`, delays) holds the forced tokens, and user audio starts.

`build(...)` returns, for one session:
  emb [P, dim]  -- the transformer input of every prompt step, in order (voice embeddings verbatim; token frames
                   embedded by our MoshiLM), so the prompt is one teacher-forced pass through MoshiFrame with emb_in;
  ring [K, CT]  -- the host token ring (LMGenHost layout, -2 = unset) after the prompt, and offset = P + 1.
"""
from __future__ import annotations

import os
import tarfile

import torch

SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
ZERO_TEXT = 3
SILENCE_FRAMES = 6  # int(0.5 * 12.5)
DEFAULT_PERSONA = ("You are a wise and friendly teacher. Answer questions or provide advice in a clear and "
                   "engaging way.")


def wrap_with_system_tags(text: str) -> str:
    t = text.strip()
    if t.startswith("<system>") and t.endswith("<system>"):
        return t
    return f"<system> {t} <system>"


_voice_cache: dict[str, dict] = {}


def load_voice(name: str, repo: str = "nvidia/personaplex-7b-v1") -> dict:
    """voices/<name>.pt from the repo's voices.tgz -> {"embeddings": [P,1,1,dim] or [P,..,dim], "cache": [1,K,CT]}"""
    if name in _voice_cache:
        return _voice_cache[name]
    if os.path.isfile(name):
        path = name
    else:
        from huggingface_hub import hf_hub_download
        tgz = hf_hub_download(repo, "voices.tgz")
        d = os.path.join(os.path.dirname(tgz), "voices_x")
        base = name if name.endswith(".pt") else name + ".pt"
        path = os.path.join(d, "voices", base)
        if not os.path.isfile(path):
            # several TP ranks may get here at once: extract into a private dir, then atomically rename into place
            import shutil
            import uuid
            tmp = f"{d}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            os.makedirs(tmp, exist_ok=True)
            with tarfile.open(tgz) as t:
                t.extractall(tmp)
            try:
                os.rename(tmp, d)
            except OSError:          # another rank won the race; its copy is complete
                shutil.rmtree(tmp, ignore_errors=True)
    st = torch.load(path, map_location="cpu")
    _voice_cache[name] = st
    return st


class PromptBuilder:
    """Builds the PersonaPlex prompt for one session.

    After the prompt, the reference's token ring is uniform: every column holds the last silence frame
    (text 3, agent = SILENCE, user = SINE) -- verified against the CPU reference (prompt_cache) -- so the host ring
    of a row is simply filled with that frame, and the row's offset continues at P + 1 (P transformer steps).
    The transformer inputs of the prompt are: the voice prompt's stored embeddings (verbatim), then the token
    frames' embeddings, where the input token of frame f is the frame forced one step earlier (delays 0 / 1 resolve
    to the previous frame's tokens; only the very first token frame sees the voice prompt's ring, which is the same
    uniform silence/sine/3 pattern)."""

    def __init__(self, lm, cfg, tokenizer):
        self.lm, self.cfg, self.tok = lm, cfg, tokenizer

    def token_frames(self, persona: str):
        fr = []
        sil = [ZERO_TEXT] + SILENCE_TOKENS + SINE_TOKENS
        fr += [sil] * SILENCE_FRAMES
        if persona:
            fr += [[int(t)] + SILENCE_TOKENS + SINE_TOKENS for t in self.tok.encode(wrap_with_system_tags(persona))]
        fr += [sil] * SILENCE_FRAMES
        return torch.tensor(fr, dtype=torch.long)                              # [Pt, 17]

    @torch.no_grad()
    def build(self, voice: str, persona: str, delays=None):
        """-> (emb [P, dim] CPU in the LM dtype, last_frame [17] (ring fill after the prompt), P).
        Replays LMGen.prepare_step_input on the reference ring (CT = cache width of the voice .pt, offset continues
        after the voice prompt) so the first token frames read the voice prompt's trailing agent codes exactly."""
        cfg = self.cfg
        delays = list(cfg.delays) if delays is None else list(delays)
        vs = load_voice(voice)
        vemb = vs["embeddings"].reshape(-1, vs["embeddings"].shape[-1])       # [Pv, dim]
        cache = vs["cache"][0].clone().long()                                   # [K, CT]
        CT = cache.shape[-1]
        offset = vemb.shape[0] + 1   # step_embeddings: offset 0 seeds, then one transformer step per embedding
        fr = self.token_frames(persona)
        inputs = []
        for f in fr.tolist():
            for k in range(cfg.num_codebooks):
                cache[k, (offset + delays[k]) % CT] = f[k]
            inputs.append(cache[:, (offset - 1) % CT].clone())
            offset += 1
        temb = self.lm.embed_tokens(torch.stack(inputs))
        emb = torch.cat([vemb.to(temb.dtype), temb], 0)
        # ring after the prompt: all streams forced, so the next user step reads column (offset - 1)
        return emb, cache, int(emb.shape[0]), offset
