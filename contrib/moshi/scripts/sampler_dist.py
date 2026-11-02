"""Distribution check: inverse-CDF top-k multinomial (device) vs torch.multinomial on softmax(topk(l/T)) (CPU)."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.lm import sample
torch.manual_seed(0)
l = torch.randn(1, 2048) * 2
N = 200000
u = torch.rand(N, 1)
got = sample(l.expand(N, -1), u, 0.8, 250)
vals, idx = torch.topk(l / 0.8, 250); p = torch.softmax(vals, -1)[0]
ref = torch.zeros(2048); ref[idx[0]] = p
emp = torch.bincount(got, minlength=2048).float() / N
print("TV distance", float(0.5 * (emp - ref).abs().sum()), "(sampling noise ~", float(0.5 * (ref * (1 - ref) / N).sqrt().sum() * 0.8), ")")
