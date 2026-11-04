# Qwen3-TTS (12Hz) on AWS Neuron — contrib

Step-by-step guide with results: [NeuronStuff/qwen3-tts-vllm-omni-neuron](https://github.com/jimburtoft/NeuronStuff/tree/main/qwen3-tts-vllm-omni-neuron).

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
| 1 server, 2 talkers (cores 0,2) + 2 vocoders (cores 1,3) | 32 | 143.8 | 5.4 s |
| 1 server, 3 talkers + 1 vocoder | 48 | 111.5 (conc 160) / 132.6 (conc 300) | 4.6 / 7.8 s |
| **1 server, 2 talkers + 2 vocoders** | **48** | **150.7 / 153.7 / 158.2** at concurrency 128 / 300 / 400 | 3.0 / 7.4 / 9.8 s |

The NKI vocoder (default on device) is on in all rows from B=32 down. Streaming, B=48 2+2: TTFA p50 59 ms at
concurrency 1, 118 ms at 64; 158.9x at concurrency 160. Single stream: 0.74 s per ~4 s utterance with a B=1 talker.
**Best throughput: LNC=1 with two independent servers.** At LNC=1 (`NEURON_LOGICAL_NC_CONFIG=1`, 8 logical cores of
half size, `QTTS_NC_COUNT=8`), a B=48 talker frame takes about the same time on half the hardware (30.6 vs 30.3 ms),
so the device yields ~2x the talker frames. One big server cannot use that (rows stay half full), but independent
servers can:

| layout (whole trn2.3xlarge) | RTFx online | p50 |
|---|---|---|
| LNC=2, 1 server, 2T + 2V (B=48) | 150-162 | 3.0-9.4 s |
| LNC=1, 1 server, 4T + 4V | 129-141 | 3.7-9.8 s |
| LNC=1, 2 servers x (2T + 2V), default host threads | 173-195 | 3.2-7.8 s |
| LNC=1, 2 servers x (2T + 2V), `QTTS_HOST_THREADS=1 QTTS_PREFILL_CTX_BUCKETS=64,128` | 248 (conc 256) / 250 (conc 400) | 3.7 / 6.0 s |
| LNC=1, same, `QTTS_SLOT_LEN=1024`, `QTTS_B=80` (upstream bench tool) | 309 (conc 400) / 318 (conc 640) | 5.1 / 8.3 s |
| LNC=1, same + 2-row prefill (default at LNC=1, B>=64) | 327-331 (conc 400) | 4.7 s |
| **LNC=1, same + async output (default)** | **365-373 (conc 400)**, 349 (conc 600), 262 (conc 200, TTFP 260 ms) | **4.2-4.3 s** |
| LNC=2, 1 server 2T B=48 + 2V, async output (default) | 176 (conc 128) / 186 (conc 256) | 2.8 / 5.7 s |
| LNC=1, 4 servers x (1T + 1V), same env | 244-247 | 3.8-9.4 s |

Streaming, LNC=1 2 servers: 265x at concurrency 256 (TTFA p50 0.93 s); 101x at concurrency 64 per server (TTFA 133 ms).
Single stream at LNC=1 (B=1 talker): 0.97 s per ~4 s utterance, TTFA 49 ms (LNC=2: 0.72 s / 41 ms).
Quality: no measurable difference between LNC=1 and LNC=2 (3 x 100 prompts each, sampled; mean WER/CER 1.03 % vs
1.15 % excluding one sampled run-on, paired z = 0.86). Recipe:

```bash
export NEURON_LOGICAL_NC_CONFIG=1   # implies QTTS_HOST_THREADS=1 and QTTS_PREFILL_CTX_BUCKETS=64,128 by default
QTTS_SLOT_LEN=1024 QTTS_B=80 QTTS_REP0=2 QTTS_DEV0=0,2 QTTS_REP1=2 QTTS_DEV1=1,3 QTTS_B1=160 python -m qwen3_tts_neuron.serving.make_deploy /tmp/a.yaml
QTTS_SLOT_LEN=1024 QTTS_B=80 QTTS_REP0=2 QTTS_DEV0=4,6 QTTS_REP1=2 QTTS_DEV1=5,7 QTTS_B1=160 python -m qwen3_tts_neuron.serving.make_deploy /tmp/b.yaml
QTTS_NC_COUNT=8 python -m qwen3_tts_neuron.serving.serve_launcher serve <model> --deploy-config /tmp/a.yaml --port 8101 ... &
QTTS_NC_COUNT=8 python -m qwen3_tts_neuron.serving.serve_launcher serve <model> --deploy-config /tmp/b.yaml --port 8102 ... &
# put a load balancer in front of 8101/8102
```
`QTTS_SLOT_LEN=1024` caps a request at 1024 positions (prompt + ~80 s of audio). Half the KV cache leaves room for
B=80 talker slots per half-size core; at B=48 it alone measured +3 % (264 -> 271x), B=80 then +14 %.

Why the two env settings matter at LNC=1:
- 8 stage processes plus 2 API servers share 12 vCPUs, so 1 torch thread per stage is enough.
- At LNC=1 the full-cache 64-token prefill is 14 ms vs 11 ms ctx-limited.
- Together they took the 2-server layout from 195x to 250x. At LNC=2 they are worth ~3 % (157 -> 162x).
- The sampler's top-k compare is miscompiled at LNC=1, so `fused.sample_rows` switches to an equivalent form there
  (identical draws for the same noise).

3 talkers + 1 vocoder is limited by the single stage-1 replica's per-step loop (requests wait ~1 s there after the
talker finishes; vocoder compute ~40 % busy), not by the device. Note: clients must allow enough connections (httpx defaults to 100, which caps concurrency silently).

```bash
QTTS_B=48 QTTS_REP0=2 QTTS_DEV0=0,2 QTTS_REP1=2 QTTS_DEV1=1,3 QTTS_B1=96 python -m qwen3_tts_neuron.serving.make_deploy /tmp/qtts.yaml
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
| `QTTS_HOST_THREADS` | 2 (1 at LNC=1) | torch host threads per stage process |
| `QTTS_ASYNC_OUT` | 1 on device | `sample_tokens` returns an async output so the device->host copy of a frame overlaps vLLM's scheduling of the next step (sets `async_scheduling: true` on the talker stage) |
| `QTTS_PREFILL_ROWS` | 2 at LNC=1 with B>=64, else off | new requests prefilled in pairs (4+ rows is slower) |
| `QTTS_C2W_BATCH` | 4 | full-context vocoder windows decoded per call across requests (exact in fp32) |
| `QTTS_C2W_NKI_BLOCKS` | `1,2,3` on device (off in CPU mode) | decoder blocks run as fused NKI kernels (`1,2,3` recommended; sharded over the LNC=2 cores: vocoder window 19.2 -> 7.0 ms at B=1, 12.5 -> 4.9 ms/window at B=4; same accuracy) |
| `QTTS_CP_NKI` | `0` | experimental: each code-predictor step (5 layers) as one NKI launch (`kernels/cp_nki.py`). Per step the kernel matches the compiled torch layers (~1.05 vs 1.03 ms at B=80, LNC=1), but the full frame is slower (B=80: 20.8 vs 19.5 ms; B=8: 15.3 vs 10.3 ms). Leave off |
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
