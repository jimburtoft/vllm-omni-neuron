# Qwen3-TTS (12Hz) on AWS Neuron — contrib

Status: **experimental contrib**. Runs the Qwen3-TTS 12Hz 1.7B checkpoints (`-VoiceDesign`, `-CustomVoice`, `-Base`)
on Trainium2 and serves them through
**vLLM-Omni** (`vllm-omni serve --omni`, OpenAI `/v1/audio/speech`, streaming) on the public SDK 2.33 DLAMI.

vLLM-Omni-Neuron 0.29.0.0.2.0 implements diffusion workers only; autoregressive omni stages (`LLM_AR`,
`LLM_GENERATION`) raise `NotImplementedError` in `get_omni_ar_worker_cls()`. This contrib supplies thin Neuron workers for
the two Qwen3-TTS stages that are selected per stage with `worker_cls:` in the deploy YAML. The stock vLLM-Omni
scheduler, chunk transfer adapter, stage-input processor, connector and OpenAI server are used unchanged.

## What runs on Neuron

| stage | component | implementation |
|---|---|---|
| 0 talker | 28-layer Qwen3 decoder (static per-slot KV) + 5-layer code predictor (15 residual codebooks) + on-device sampling | **one compiled graph per audio frame** for B streams (`fused.py`) |
| 1 code2wav | RVQ dequant + 8-layer sliding-window transformer + SnakeBeta conv-transpose upsampler (12.5 Hz → 24 kHz) | windowed graphs, exact left-context cropping (`code2wav.py`) |

Graphs are plain PyTorch compiled with `torch.compile` — torch-neuronx (`backend="neuron"`) in the torch-neuronx venv,
libtorch-neuronx-lite's native backend inside the vLLM-Omni venv. fp32 CPU runs match the reference `qwen_tts` package
exactly (`tests/`).

## Requirements
- trn2 instance, Deep Learning AMI Neuron SDK 2.33 (Ubuntu 24.04).
- vLLM-Omni venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0` (vllm 0.29.0, vllm-omni 0.29.0rc1).
- Optional bare-engine venv: `/opt/aws_neuronx_venv_torch_neuronx_2_14_0`.

Activate the venv (`source .../bin/activate`) rather than calling its `python` directly: the compiler is found
through `PATH` (otherwise compilation fails with `posix_spawn failed: No such file or directory`).

```bash
git clone -b contrib/qwen3-tts https://github.com/jimburtoft/vllm-omni-neuron.git
export PYTHONPATH=$PWD/vllm-omni-neuron/contrib/qwen3_tts NEURON_SKIP_EFA_AFFINITY=1
```

## Online server (OpenAI-compatible)

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
python -m qwen3_tts_neuron.serving.make_deploy /tmp/qtts.yaml          # talker on core 0, code2wav on core 1, B=8
QTTS_NC_COUNT=4 python -m qwen3_tts_neuron.serving.serve_launcher serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --deploy-config /tmp/qtts.yaml --port 8091 --trust-remote-code --omni \
    --init-timeout 3600 --stage-init-timeout 3600

curl -s localhost:8091/v1/audio/speech -H 'Content-Type: application/json' -o out.wav -d '{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "task_type": "VoiceDesign", "response_format": "wav",
  "input": "She said she would be here by noon.", "language": "English",
  "instructions": "Calm, warm adult female voice."}'
```

The first start on an instance compiles every graph (~25-30 min at B=8; later starts take ~3 min from the compile
cache), and the default `--init-timeout` (600 s) is too short for that first start.

Streaming: add `"stream": true, "stream_format": "audio", "response_format": "pcm"` (16-bit PCM, 24 kHz).
Client with concurrency and streaming: `python -m qwen3_tts_neuron.serving.speech_client <base> <out_dir> <conc> <n>`.

A trn2.3xlarge has 4 logical cores (LNC=2). Layouts measured with the online server (VoiceDesign, sampled, ~4 s
utterances; RTFx = seconds of audio per second for the whole device):

| layout | talker B | RTFx | p50 latency |
|---|---|---|---|
| 2 servers x (1 talker + 1 vocoder) | 8 | 42.9 | 4.3 s |
| 1 server, 2 talkers (cores 0,2) + 1 shared vocoder (core 1) | 8 | 44.0 | 2.8 s |
| 1 server, 3 talkers (cores 0,2,3) + 1 shared vocoder (core 1) | 16 | 61.5 | 5.6 s |
| **1 server, 2 talkers (cores 0,2) + 2 vocoders (cores 1,3)** | **32** | **119.0** | 5.0 s |
| 1 server, 3 talkers + 1 vocoder | 32 | 61.5 | 9.8 s (vocoder-bound) |

The B=32 row is at concurrency 160; at concurrency 96 the same server gives 117x at p50 2.9 s. Streaming TTFA p50 is
118 ms at concurrency 32. For latency, use a B=1 talker: 0.74 s per ~4 s utterance, TTFA 44 ms.

```bash
QTTS_B=32 QTTS_REP0=2 QTTS_DEV0=0,2 QTTS_REP1=2 QTTS_DEV1=1,3 QTTS_B1=64 python -m qwen3_tts_neuron.serving.make_deploy /tmp/qtts.yaml
```
Each stage process caps torch host threads at `QTTS_HOST_THREADS` (default 2). The default thread pool spins ~3
cores per process and starved the other stages on the 12-vCPU host (3 talkers + vocoder: 35x before the cap, 61x
after).
A replicated stage must list one core per replica. A single core with `num_replicas > 1` is read by vLLM-Omni as a
per-replica template, so replica r would land on core r; `make_deploy` rejects that. For two independent servers,
generate a second YAML with `QTTS_DEV0=2 QTTS_DEV1=3` and start it on another port with the same launcher
(`serve_launcher` keeps the API-server process from claiming the Neuron runtime).

`make_deploy` environment:

| var | default | meaning |
|---|---|---|
| `QTTS_B` | 8 | talker batch slots (= stage-0 `max_num_seqs`); 32 is the throughput setting |
| `QTTS_DEV0` / `QTTS_DEV1` | 0 / 1 | logical cores for talker / code2wav (one per replica, comma-separated) |
| `QTTS_REP0` / `QTTS_REP1` | 1 / 1 | replicas of the talker / code2wav stage |
| `QTTS_SLOT_LEN` | 2048 | per-request KV slots = stage-0 `max_model_len` (prompt + frames; 12.5 frames per second of audio, so ~2.5 min) |
| `QTTS_PREFILL_CTX` / `QTTS_PREFILL_CTX_BUCKETS` | 256 / 128 | prefill buckets that attend only the first N KV slots when the prompt fits (exact; Tb=128 prefill 65 -> 16 ms at B=32) |
| `QTTS_HOST_THREADS` | 2 | torch host threads per stage process |
| `QTTS_CP_VEC` | 0 | `1` = elementwise code-predictor attention (slower under lite: 24 vs 10 ms at B=32) |
| `QTTS_GREEDY` | 0 | `1` writes greedy defaults into the YAML |

Sampling is **per request** (one compiled graph serves mixed batches). Defaults come from the deploy YAML
(`default_sampling_params` for the talker, `subtalker_sampling_params` for the code predictor; checkpoint defaults
T=0.9, top-k 50 for both; `QTTS_GREEDY=1` writes greedy defaults). Per request over HTTP:

```json
{"seed": 7, "max_new_tokens": 600,
 "extra_params": {"temperature": 0.9, "top_k": 50,
                  "subtalker_dosample": true, "subtalker_temperature": 0.9, "subtalker_top_k": 50}}
```
`temperature: 0` gives greedy. A request with a `seed` is reproducible regardless of what else is in the batch.
top-k is capped at 64.

## Task types

All three checkpoints use the same graphs. Talker prompts are built on the host by vLLM-Omni's own
`Qwen3TTSPromptEmbedsBuilder` (`serving/prompts.py`), so request fields match the GPU path:

| checkpoint | `task_type` | fields |
|---|---|---|
| `-VoiceDesign` | `VoiceDesign` | `instructions` (voice description) |
| `-CustomVoice` | `CustomVoice` | `voice` (preset speaker: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian), optional `instructions` |
| `-Base` | `Base` | `ref_audio` (URL / base64 / path) + `ref_text` for in-context cloning; `x_vector_only_mode: true` for speaker embedding only |

Base checkpoints also load the speaker encoder and the speech-tokenizer encoder. Both run on CPU, once per reference
clip (results are cached by clip hash). Streaming-text mode (`non_streaming_mode: false`, the default for Base) is
supported: the remaining text embeddings are fed one per frame.

## Offline (vLLM-Omni `Omni` API)
```bash
python -m qwen3_tts_neuron.serving.run_omni /tmp/qtts.yaml out/ 4
```

## Bare engine (no vLLM)
```python
from qwen3_tts_neuron.pipeline import Qwen3TTSNeuron
tts = Qwen3TTSNeuron("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", batch_size=8)
wavs, sr = tts.generate_voice_design(["Hello there."], instruct=["Calm female voice."], language=["English"])
```

## Tests
`pytest tests/` (CPU, fp32; the equivalence tests need the `qwen-tts` package for the reference).

## Known limitations
- The reference encoders for Base voice cloning run on CPU (about 0.1-0.5 s per new reference clip).
- Greedy decoding with CustomVoice + `instructions` can loop until `max_new_tokens`; the HF reference does the same.
  Use the sampled defaults.
- `top_p` is ignored (checkpoint default is 1.0); top-k ≤ 64.
- **Long text**: prompts are prefilled in chunks (any length up to the slot), and decode attends only the used KV
  range (context buckets `QTTS_CTX_BUCKETS`, default 256/512/1024/2048). Prompts above `QTTS_MAX_PROMPT` (400 positions,
  ~1,000 characters of English) are rejected (HTTP 400 "did not produce audio"). The checkpoint itself degrades on very
  long single utterances (HF reference: ~500-token prompts produce unrelated speech; greedy decoding repeats on long
  text, sampled decoding is fine). Split long text into sentences/paragraphs (the WebSocket `/v1/audio/speech/stream`
  endpoint has `split_granularity: "sentence"`).
- vllm-omni 0.29.0rc1's `talker2code2wav_async_chunk` re-sends the final chunk when an utterance ends exactly on a
  25-frame boundary (fixed upstream on main). `qwen3_tts_neuron.serving.backports` applies the same fix at worker
  start when the installed version lacks it.
