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

## Results (trn2.3xlarge, SDK 2.33, one stage-0 core)

Per-frame step with encoder, LM and decoder all on one core (budget 80 ms):

| Sessions | step p50 | of which LM |
|---|---|---|
| 1 | 54 ms | 44 ms |
| 2 | 71 ms | 56 ms |
| 3 | 81 ms | 66 ms |

Served (`/v1/realtime?duplex=1`, stage 0 on core 0, decoder on core 1), audio streamed at real-time pace:

| Concurrent sessions | Frames returned | First audio | Lag vs real time p50 / p99 |
|---|---|---|---|
| 1 | all | 152 ms | 184 / 274 ms |
| 3 | all | 190-260 ms | 250 / 381 ms |

Two servers (cores 0+1, 2+3) give 6 real-time sessions per trn2.3xlarge with these first-version graphs (plain
PyTorch, no custom kernels).

Output quality: the greedy Neuron output matches the kyutai reference (fp32, CPU) text exactly on our test clips and
Whisper transcribes the generated speech identically to Moshi's inner-monologue text.

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
MOSHI_B=3 python -m moshi_neuron.serving.make_deploy /tmp/moshi.yaml   # 3 sessions; stage 0 on core 0, stage 1 on core 1
MOSHI_NC_COUNT=4 python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
    --deploy-config /tmp/moshi.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600
```

The first start compiles the graphs (about 8 minutes); later starts load them from the compile cache.

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
