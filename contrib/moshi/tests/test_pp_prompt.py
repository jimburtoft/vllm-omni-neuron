"""PromptBuilder vs the NVIDIA reference's recorded prompt transformer inputs (pp goldens, prompt_emb [P, dim])."""
import os
import sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    from vllm_neuron.vllm.platform import NeuronPlatform
    NeuronPlatform.set_device_count(4)
    from safetensors.torch import load_file
    import sentencepiece
    from huggingface_hub import hf_hub_download
    from moshi_neuron.prompt import PromptBuilder
    from moshi_neuron.config import LMConfig
    g = torch.load(sys.argv[1])
    sd = load_file(os.environ["MOSHI_WEIGHTS"])

    class E:  # minimal embedder with our MoshiLM table layout
        def __init__(self):
            cfg = LMConfig()
            tabs = [sd["text_emb.weight"]] + [sd[f"emb.{k}.weight"] for k in range(cfg.n_q)]
            self.emb = torch.cat(tabs, 0).float()
            offs, o = [], 0
            for t in tabs:
                offs.append(o); o += t.shape[0]
            self.emb_off = torch.tensor(offs).view(1, -1)

        def embed_tokens(self, tokens):
            valid = (tokens >= 0).float().unsqueeze(-1)
            return (torch.nn.functional.embedding(tokens.clamp(min=0) + self.emb_off, self.emb) * valid).sum(1)
    tok = sentencepiece.SentencePieceProcessor(hf_hub_download("nvidia/personaplex-7b-v1", "tokenizer_spm_32k_3.model"))
    pb = PromptBuilder(E(), LMConfig(), tok)
    emb, ring, P, off = pb.build(g["voice"], g["prompt"])
    ref = g["prompt_emb"].float()
    print("P ours", P, "ref", ref.shape[0], "offset ref", g["prompt_offset"])
    n = min(P, ref.shape[0])
    rel = ((emb[:n].float() - ref[:n]).norm(dim=-1) / ref[:n].norm(dim=-1))
    print("rel err per step: max %.2e mean %.2e; worst step %d" % (rel.max(), rel.mean(), int(rel.argmax())))
    print("ring equal to reference:", torch.equal(ring, g["prompt_cache"][0].long()), "offset", off, "ref", g["prompt_offset"])
