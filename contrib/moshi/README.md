# Moshi (kyutai) on AWS Trainium2 with vLLM-Omni

[Moshi](https://github.com/kyutai-labs/moshi) is a full-duplex speech-to-speech model. It listens and speaks at the
same time, in 80 ms frames: a Mimi codec encodes the user's audio, a 7B temporal transformer (Helium) plus a small
depth transformer generate Moshi's text and audio tokens, and Mimi decodes them to 24 kHz speech.

This contrib serves `kyutai/moshiko-pytorch-bf16` (or `kyutai/moshika-pytorch-bf16`) and NVIDIA
`nvidia/personaplex-7b-v1` (same architecture, with voice + persona system prompt) on Trainium2 through
**vLLM-Omni's full-duplex realtime API** (`/v1/realtime?duplex=1`), using the PersonaPlex pipeline that ships in
vllm-omni 0.29 and two Neuron workers from this directory. It runs on the public SDK 2.33 DLAMI without upstream
source changes.

| Piece | Where it runs | What it is |
|---|---|---|
| Stage 0 | 1 NeuronCore | Mimi encoder + one fused per-frame graph (Helium step, 8-step depth transformer, on-device sampling) for B sessions |
| Stage 1 | 1 NeuronCore | Streaming Mimi decoder for B sessions |
| Duplex front end | host | vLLM-Omni realtime WebSocket server (unchanged) |

All streaming state (Mimi convolution buffers and transformer KV caches, the 3000-frame Helium KV) lives in
device memory per session row; only 1920 PCM samples and ~10 tokens cross the host boundary per frame.

## Results (one trn2.3xlarge, SDK 2.33)

**Highest capacity: PersonaPlex, TP4 throughput layout, packed FP8 KV cache** (`nvidia/personaplex-7b-v1`, voice
prompt + persona, 1500-frame attention window as in the PersonaPlex reference; sessions arriving 0.5-1 s apart):

| Session length | Concurrent sessions | Every frame on time | Per-frame latency p50 / p99 |
|---|---|---|---|
| 4 minutes | **36** | yes | 39-51 / 230 ms |
| 60 s | 36 | yes | 54-68 / 1275 ms |
| 60 s | 40 | 740 / 750 frames | 1.0-1.2 s (output path queues) |
| 4 minutes | 40 | no (2727 / 3000 frames) | 7.8-10.8 s |

The Neuron side keeps up with more: in steady state stage 0 processes exactly the required frames per second
for 40, 44 and 48 sessions (stage-0 step ~51 ms at 40 rows). Above ~36-40 sessions the limit is the vLLM-Omni API
server process (WebSocket frontend and orchestrator share one Python GIL), not the device; the vLLM-Omni
multi-process orchestrator / multi-frontend work upstream addresses this.

**moshiko / moshika** (3000-frame window), served, audio streamed at real-time pace, 28.8 s conversation:

| Layout | Concurrent sessions | Every frame on time | First audio | Per-frame latency p50 / p99 |
|---|---|---|---|---|
| 1 server (cores 0+1), B=1 | 1 | yes | 175 ms | 93 / 96 ms |
| 1 server, B=4 | 4 | yes | 220-230 ms | 112-129 / 151-167 ms |
| 1 server, B=5 | 5 | yes | 242 ms | 135-147 / 176-186 ms |
| 2 servers (cores 0+1, 2+3), B=5 each | 10 | yes | 190-260 ms | 130-143 / <= 180 ms |
| **1 server, 3 stage-0 replicas (cores 0,2,3) B=5 + 1 decoder (core 1) -- latency layout** | **15** | yes | <= 480 ms | 83-118 / <= 193 ms |
| same, 1 / 3 sessions | 1 / 3 | yes | | **54 / 60** ; 72-74 / 105 ms |
| **TP4 over all cores, codec on the TP ranks -- throughput layout** | **24** | yes | <= 600 ms | 194-214 / 401-539 ms |
| same, 1 / 16 sessions | 1 / 16 | yes | | **44 / 83** ; 106-134 / 202 ms |

Per-frame latency is the time from sending an 80 ms input frame to receiving the 80 ms of Moshi audio it
produces; it does not grow over the session (also measured over 5-minute sessions). B=6 per server does not fit one core's HBM (7B weights + 6 x 1.6 GB KV).

The Helium attention runs on the fused nkilib decode kernel (`attention_block_tkg`: RMSNorm + QKV + RoPE +
attention + KV write + output projection in one kernel) over a block-paged ring KV cache with an in-kernel
sliding-window mask; the rest is plain PyTorch compiled by the Neuron compiler.

Output quality: greedy decoding reproduces the kyutai reference's text on our test clips; teacher-forced against
the fp32 reference, text predictions match 100% and audio-code predictions 99.5%; Whisper's transcript of the generated
speech matches Moshi's inner-monologue text.

Sessions can run past Moshi's 3000-frame (4 min) attention window: the KV caches are rings, and the server allows
sessions up to 1 hour (`MOSHI_MAX_MODEL_LEN`, frames). Note that Moshi itself tends to repeat or go quiet in very
long one-sided monologues; the kyutai reference does the same.

## Setup

```bash
# trn2.3xlarge, "Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0"
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
git clone -b contrib/moshi https://github.com/jimburtoft/vllm-omni-neuron.git
export PYTHONPATH=$PWD/vllm-omni-neuron/contrib/moshi NEURON_SKIP_EFA_AFFINITY=1
```

Add swap before the first compile (`sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile &&
sudo mkswap --force /swapfile && sudo swapon /swapfile`).

## Serve

```bash
MOSHI_B=5 python -m moshi_neuron.serving.make_deploy /tmp/moshi.yaml   # 5 sessions; stage 0 on core 0, stage 1 on core 1
MOSHI_NC_COUNT=4 python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
    --deploy-config /tmp/moshi.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600
```

The first start compiles the graphs (about 8 minutes); later starts load them from the compile cache.

### 15 sessions per instance

The Mimi decoder is light, so one decoder core can serve three language-model cores:

```bash
MOSHI_B=5 MOSHI_REP0=3 MOSHI_DEV0=0,2,3 MOSHI_DEV1=1 python -m moshi_neuron.serving.make_deploy /tmp/moshi15.yaml
MOSHI_MLP_XW=1 MOSHI_BUCKETS=1,2,3,4,5 MOSHI_DEC_BUCKETS=1,2,4,8,15 MOSHI_CTX_BUCKETS=512,1024,2048 MOSHI_NC_COUNT=4 python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
    --deploy-config /tmp/moshi15.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600
```

The first start compiles the three stage-0 replicas one after another (about 20 minutes); later starts use the cache.

### Throughput layout: 24 sessions per instance (tensor parallel over all 4 cores)

Helium sharded over the 4 NeuronCores (TP4, in-graph all-reduce); every rank runs the Mimi encoder for all sessions
and the Mimi decoder for its quarter of the sessions; stage 1 is a CPU pass-through:

```bash
MOSHI_B=24 MOSHI_TP0=4 MOSHI_DEV0=0,1,2,3 MOSHI_DEV1=0 MOSHI_S0_DECODE=1 MOSHI_MLP_XW=1 \
    python -m moshi_neuron.serving.make_deploy /tmp/moshi_tp4.yaml
MOSHI_S0_DECODE=1 MOSHI_MLP_XW=1 MOSHI_BUCKETS=1,2,4,8,16,24 MOSHI_CTX_BUCKETS=512,1024,2048 MOSHI_NC_COUNT=4 python -m moshi_neuron.serving.serve_launcher serve \
    kyutai/moshiko-pytorch-bf16 --deploy-config /tmp/moshi_tp4.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600
```

Measured: 24 concurrent sessions, every frame, latency p50 216-242 / p99 579 ms (stage-0 step 75 ms); 20 sessions:
p50 149-196 / p99 327 ms. Teacher-forced parity on 20 rows identical to one core. `MOSHI_B` must be a multiple of 4.
(LNC=1 / TP8 was measured too: faster on device, but host overhead of 8 workers on 12 vCPUs makes it serve fewer
sessions -- not recommended.)

**Context buckets** (`MOSHI_CTX_BUCKETS`): the attention kernel reads only as many KV blocks as the longest active
session has filled (young sessions are cheap; at a full 4-minute window it falls back to all 3072 slots).

**Batch buckets** (`MOSHI_BUCKETS`, `MOSHI_DEC_BUCKETS`): every graph is compiled for each listed batch size and a
step runs the smallest one that covers the active sessions, so the step time (and latency) scales with load instead
of always paying for the maximum batch. Each bucket adds a compile on first start (cached afterwards).

### PersonaPlex, 36-40 sessions per instance (TP4, packed FP8 KV)

`nvidia/personaplex-7b-v1` is gated on Hugging Face (request access first). Serve the local snapshot directory (the
voice prompts are resolved from its `voices.tgz`):

```bash
PP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('nvidia/personaplex-7b-v1'))")
export PYTHONPATH=$PYTHONPATH:$PWD/vllm-omni-neuron/contrib/moshi/scripts/fs_site   # frame-sync gate + GC settings
export MOSHI_CONTEXT=1500 MOSHI_KV_SLOTS=1536 MOSHI_CTX_BUCKETS=512,1024 MOSHI_BUCKETS=24,32,40,48 \
  MOSHI_FP8_KV=1 MOSHI_FP8_PACKED=1 MOSHI_ENC_SHARD=1 MOSHI_DEP_RUN=8 \
  MOSHI_EMB_INPUTS=1 MOSHI_PP_PREFILL=1 \
  MOSHI_FRAMESYNC=1 MOSHI_FS_PERIOD_MS=60 MOSHI_WORKER_THREADS=1 MOSHI_GC_FULL=0 \
  MOSHI_FAST_APPEND=1 MOSHI_CONS_FIX=1 MOSHI_MARKS_FIX=1 MOSHI_WS_NO_DEFLATE=1 \
  MOSHI_STAGE_EVENTS_CAP=64 MOSHI_MARK_COALESCE=1 MOSHI_SNAP_FIX=1 \
  MOSHI_PCM_FAST=1 MOSHI_SHM_PROBE=1 MOSHI_API_TORCH_THREADS=4 \
  MOSHI_S0_DECODE=1 MOSHI_MLP_XW=1 MOSHI_NC_COUNT=4
MOSHI_B=48 MOSHI_TP0=4 MOSHI_DEV0=0,1,2,3 MOSHI_DEV1=0 python -m moshi_neuron.serving.make_deploy /tmp/pp_tp4.yaml
python -m moshi_neuron.serving.serve_launcher serve $PP --deploy-config /tmp/pp_tp4.yaml --port 8000 --omni \
    --init-timeout 10800 --stage-init-timeout 10800
```

What the main switches do:
- `MOSHI_FP8_KV=1 MOSHI_FP8_PACKED=1`: FP8 (e4m3) KV cache in the attention kernel's packed layout, which lets the
  kernel load K with a DMA transpose (attention -41% at 40 sessions vs unpacked FP8, bit-identical output).
  FP8 KV costs ~1.4 points of teacher-forced audio-code top-1 vs bf16 KV; text is unchanged.
- `MOSHI_PP_PREFILL=1`: the voice + persona prompt of a new session runs as one batched prefill (19 ms), so first
  audio arrives in 0.1-0.5 s and other sessions are not stalled.
- `MOSHI_FRAMESYNC=1 MOSHI_FS_PERIOD_MS=60`: stage 0 waits (up to 60 ms) until the sessions' next frames have
  arrived, so each step carries close to all active sessions instead of splitting a tick into partial batches.
- `MOSHI_DEP_RUN=8`: only the agent-stream depth-transformer steps run (PersonaPlex also predicts the user stream).
- `MOSHI_GC_FULL=0`: no automatic gen-2 garbage collection (its pauses grow to ~1 s over a 4-minute session).

For calls longer than about 2 minutes, keep the system prompt in context with pinning:
`MOSHI_PP_PIN=160 MOSHI_KV_SLOTS=2048 VLLM_NEURON_ATTN_MASK_TORCH=1` and no `MOSHI_CTX_BUCKETS`. This was measured
with unpacked FP8 at 32 sessions x 4 min (p99 352 ms); sessions stay in persona to the end.

The first start compiles every batch x context bucket graph (about 2 hours); later starts use the cache.

## Talk to it

```bash
python -m moshi_neuron.serving.duplex_client --wav my_question.wav --sessions 1 --out out/
```

The client also records Moshi's inner-monologue text (`out/session0.txt`), streamed as `text` deltas on
`response.output_audio.delta` events. The client streams the file in real time over `ws://127.0.0.1:8000/v1/realtime?duplex=1` (24 kHz float32 in,
16-bit PCM out) and writes Moshi's reply to `out/session0.wav`. Any client of the vLLM-Omni realtime duplex API
works (`vllm_omni.clients.duplex.DuplexClient`). Moshi takes no `voice` or `instructions`; for PersonaPlex pass
`--voice NATM1.pt --instructions "<persona text>"` (and `--stagger-s 0.5` to spread session arrivals in a load test).

## Standalone (no server)

```bash
python contrib/moshi/scripts/run_e2e.py --goldens <dir with clip_*.npy> --clips utt1 --B 1 --greedy
```

## Files

- `moshi_neuron/mimi.py`: streaming Mimi encoder/decoder graphs (state in device buffers, per-row reset/active).
- `moshi_neuron/lm.py`: Helium + depth transformer frame graph, on-device Gumbel top-k sampling, host delay ring.
- `moshi_neuron/engine.py`: stage-0 / decoder halves and the standalone engine.
- `moshi_neuron/serving/`: vLLM-Omni workers, deploy generator, launcher, duplex client, 0.29 backport
  (Moshi sessions open without the PersonaPlex voice/persona prefill, as upstream vllm-omni PR #9319 does on main).
- `tests/`: CPU and Neuron parity against the kyutai reference.

## License

Code: Apache-2.0 (this repo). Moshi / Mimi weights: CC-BY 4.0 (kyutai).
