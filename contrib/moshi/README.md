# Moshi (kyutai) on AWS Trainium2 with vLLM-Omni

[Moshi](https://github.com/kyutai-labs/moshi) is a full-duplex speech-to-speech model. It listens and speaks at the
same time, in 80 ms frames: a Mimi codec encodes the user's audio, a 7B temporal transformer (Helium) plus a small
depth transformer generate Moshi's text and audio tokens, and Mimi decodes them to 24 kHz speech.

This contrib serves `kyutai/moshiko-pytorch-bf16` (or `kyutai/moshika-pytorch-bf16`) on Trainium2 through
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

## Results (trn2.3xlarge, SDK 2.33)

Served (`/v1/realtime?duplex=1`), audio streamed at real-time pace, 28.8 s conversation:

| Layout | Concurrent sessions | Every frame on time | First audio | Per-frame latency p50 / p99 |
|---|---|---|---|---|
| 1 server (cores 0+1), B=1 | 1 | yes | 175 ms | 93 / 96 ms |
| 1 server, B=4 | 4 | yes | 220-230 ms | 112-129 / 151-167 ms |
| 1 server, B=5 | 5 | yes | 242 ms | 135-147 / 176-186 ms |
| 2 servers (cores 0+1, 2+3), B=5 each | 10 | yes | 190-260 ms | 130-143 / <= 180 ms |
| **1 server, 3 stage-0 replicas (cores 0,2,3) B=5 + 1 decoder (core 1)** | **15** | yes | <= 280 ms | 124-140 / <= 187 ms |

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
MOSHI_NC_COUNT=4 python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
    --deploy-config /tmp/moshi15.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600
```

The first start compiles the three stage-0 replicas one after another (about 20 minutes); later starts use the cache.

## Talk to it

```bash
python -m moshi_neuron.serving.duplex_client --wav my_question.wav --sessions 1 --out out/
```

The client streams the file in real time over `ws://127.0.0.1:8000/v1/realtime?duplex=1` (24 kHz float32 in,
16-bit PCM out) and writes Moshi's reply to `out/session0.wav`. Any client of the vLLM-Omni realtime duplex API
works (`vllm_omni.clients.duplex.DuplexClient`); Moshi takes no `voice` or `instructions`.

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
