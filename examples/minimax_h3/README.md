# MiniMax-H3 on AWS Trainium2

Text-to-video-and-audio with [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) on Trainium2,
through [vLLM-Omni](https://github.com/vllm-project/vllm-omni) and the `vllm-omni-neuron` plugin.

| Configuration | Instance | NeuronCores | Typical use |
|---|---|---|---|
| **Single node, full instance** | `trn2.48xlarge` (16 Trainium2 devices) | 64 (TP=8 x CP=8) | lowest latency per video |
| **Single device** | `trn2.3xlarge` (1 Trainium2 device) | 4 (TP=4) | lowest cost per video; scale out with more instances |

Both produce 1344x768, 124 frames (5.2 s at 24 fps) with a stereo soundtrack, muxed into one `.mp4`.
The 8-step distilled mode (PDD8) uses the `alibaba-pai/MiniMax-H3-Acc-LoRAs` LoRA; the base model
runs 50 steps.

Measured warm request times (second and later requests of a running process, 29-token prompt):

| Configuration | PDD8 (8 steps) | Base (50 steps) |
|---|---:|---:|
| `trn2.48xlarge`, 64 cores | 14.2 s | 59.4 s |
| `trn2.3xlarge`, 4 cores | 130 s (up to ~285 s, see "Memory") | 765 s |

## What you need

| Item | Value |
|---|---|
| AMI | *Deep Learning AMI Neuron (Ubuntu 24.04) 20260818* (Neuron SDK 2.32). Look up the AMI ID for your region by that name. |
| Container | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04` |
| Plugin | `vllm-omni-neuron`, branch `h3-single-device` of `github.com/jimburtoft/vllm-omni-neuron` (see "Code" below) |
| Extra package | `libtorch-neuronx-lite==2.11.0.1.0.2651+723ba691` (from `pip.repos.neuron.amazonaws.com`) |
| Root volume | 600 GB gp3 (weights ~150 GB, PDD8 merged copy ~70 GB, container, compile caches) |
| Model weights | `MiniMaxAI/MiniMax-H3` (t2va files only, ~144 GB); for PDD8 also `alibaba-pai/MiniMax-H3-Acc-LoRAs` |

### Code

The MiniMax-H3 support is not yet in a released `vllm-omni-neuron`. It lives in a chain of pull
requests on a fork; the branch below contains all of them, in order:

| Pull request | Adds | Needed for |
|---|---|---|
| [whn09/vllm-omni-neuron (branch `minimax-h3`)](https://github.com/whn09/vllm-omni-neuron/tree/minimax-h3) | the MiniMax-H3 pipeline, DiT, conditioner, VAEs, ring attention | everything |
| [whn09/vllm-omni-neuron#1](https://github.com/whn09/vllm-omni-neuron/pull/1) | PDD8 heads, AdaLN on the host, single-device attention options | single device, PDD8 |
| [whn09/vllm-omni-neuron#2](https://github.com/whn09/vllm-omni-neuron/pull/2) | partial DiT eviction around the video decode | single device |
| [whn09/vllm-omni-neuron#3](https://github.com/whn09/vllm-omni-neuron/pull/3) | conditioner on the NeuronCores, streamed from host memory | single device |
| `h3-single-device` (this branch) | `run.py --single-device`, `neuron_cache.py`, this README | single device |

Use the branch as one unit; do not cherry-pick between the pull requests.

## 1. Launch the instance

Launch a `trn2.48xlarge` or `trn2.3xlarge` with the AMI above and a 600 GB gp3 root volume
(throughput 1000 MB/s speeds up the weight download and load). `trn2` instances are sold as
[EC2 Capacity Blocks for ML](https://aws.amazon.com/ec2/capacityblocks/); launch into your block with
`--instance-market-options '{"MarketType":"capacity-block"}'` and your reservation ID.

## 2. Add swap

The DLAMI has no swap, and the first-time compile needs far more host memory than the instance has
(about 150 GB on a `trn2.3xlarge`). Without enough swap the compiler is killed and the run fails with
a misleading compilation error.

```bash
# trn2.3xlarge (124 GB RAM): 192 GB of swap. A trn2.48xlarge (2 TB RAM) can skip this step.
sudo fallocate -l 192G /swapfile
sudo chmod 600 /swapfile
sudo mkswap --force /swapfile     # --force is required on the DLAMI kernel
sudo swapon /swapfile
```

## 3. Download the weights

```bash
export H3=/home/ubuntu/h3
mkdir -p $H3/out $H3/neff-cache $H3/nki-cache $H3/hf
python3 -m venv $H3/hfvenv && $H3/hfvenv/bin/pip install -q -U "huggingface_hub[hf_xet]"
$H3/hfvenv/bin/python - <<EOF
from huggingface_hub import snapshot_download
# t2va only: skip the FL2VA/Ref2VA duplicates, the ref2va transformer and the assets (~350 GB).
snapshot_download("MiniMaxAI/MiniMax-H3", local_dir="$H3/MiniMax-H3", max_workers=16,
                  ignore_patterns=["FL2VA/*", "Ref2VA/*", "transformer_ref/*", "assets/*"])
# PDD8 (optional): the 8-step LoRA.
snapshot_download("alibaba-pai/MiniMax-H3-Acc-LoRAs", local_dir="$H3/pdd",
                  allow_patterns=["MiniMax-H3-FL2VA-Acc-8Step.safetensors"])
EOF
```

## 4. Start the container and install the plugin

```bash
IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04
sudo docker pull $IMAGE
DEVICES=$(for d in /dev/neuron*; do printf -- "--device=%s " $d; done)
sudo docker run -d --name h3 $DEVICES \
  --cap-add SYS_ADMIN --cap-add IPC_LOCK --shm-size=16g --ipc=host \
  --entrypoint /bin/bash --workdir /workspace -v $H3:/workspace \
  -e VLLM_NEURON_BACKEND=neuron_native \
  -e VLLM_NEURON_LIBTORCH_NEURONX_LITE=1 \
  -e VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1 \
  -e HF_HOME=/workspace/hf \
  -e TORCH_NEURONX_NEFF_CACHE_DIR=/workspace/neff-cache \
  -e NKI_COMPILE_CACHE_URL=/workspace/nki-cache \
  $IMAGE -c 'sleep infinity'

sudo docker exec h3 bash -c '
  git clone -b h3-single-device https://github.com/jimburtoft/vllm-omni-neuron.git plugin &&
  pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e plugin &&
  pip install --no-deps --extra-index-url=https://pip.repos.neuron.amazonaws.com \
      "libtorch-neuronx-lite==2.11.0.1.0.2651+723ba691"'
```

The compiled graphs (`$H3/neff-cache`) and NKI kernels (`$H3/nki-cache`) persist on the host, so
restarting the container does not recompile. Without `TORCH_NEURONX_NEFF_CACHE_DIR` the graphs go to
`/tmp/neff_cache` inside the container and are lost with it.

## 5. (PDD8 only) Merge the LoRA

```bash
sudo docker exec h3 python plugin/examples/minimax_h3/tools/merge_fewstep_lora.py pdd \
  /workspace/MiniMax-H3 /workspace/pdd/MiniMax-H3-FL2VA-Acc-8Step.safetensors /workspace/MiniMax-H3-PDD8
```

This writes a merged `transformer/` and `pdd_heads.safetensors` to `MiniMax-H3-PDD8/` and links every
other component back to `MiniMax-H3/`, so only the transformer is copied.

## 6. Generate

### Single node, `trn2.48xlarge` (64 NeuronCores)

```bash
# Base model, 50 steps
sudo docker exec h3 bash -c 'cd plugin && python examples/minimax_h3/run.py \
  --model-path /workspace/MiniMax-H3 --tensor-parallel-size 8 --ring-degree 8 \
  --prompt "A street busker plays a bright melody on a violin under warm evening light" \
  --output /workspace/out/busker.mp4'

# PDD8, 8 steps
sudo docker exec -e MINIMAX_H3_PDD_HEADS=/workspace/MiniMax-H3-PDD8/pdd_heads.safetensors h3 bash -c '
  cd plugin && python examples/minimax_h3/run.py \
  --model-path /workspace/MiniMax-H3-PDD8 --tensor-parallel-size 8 --ring-degree 8 --steps 9 \
  --prompt "A street busker plays a bright melody on a violin under warm evening light" \
  --output /workspace/out/busker_pdd8.mp4'
```

### Single device, `trn2.3xlarge` (4 NeuronCores)

```bash
# PDD8, 8 steps
sudo docker exec -e MINIMAX_H3_PDD_HEADS=/workspace/MiniMax-H3-PDD8/pdd_heads.safetensors h3 bash -c '
  cd plugin && python examples/minimax_h3/run.py --single-device \
  --model-path /workspace/MiniMax-H3-PDD8 --steps 9 \
  --prompt "A street busker plays a bright melody on a violin under warm evening light" \
  --output /workspace/out/busker_pdd8.mp4'

# Base model, 50 steps
sudo docker exec h3 bash -c 'cd plugin && python examples/minimax_h3/run.py --single-device \
  --model-path /workspace/MiniMax-H3 \
  --prompt "A street busker plays a bright melody on a violin under warm evening light" \
  --output /workspace/out/busker.mp4'
```

`--single-device` sets TP=4 and the settings a single device needs to fit the model (listed under
"Settings" below); it is text-to-video only.

`--steps 9` with PDD8 is 9 sigmas, i.e. 8 denoising steps; the PDD heads require exactly that.

`run.py` options: `--prompt` (repeatable: prompts run in order in one process), `--repeat N` (send
each prompt N times, timed), `--seed`, `--height/--width/--num-frames` (each new size is a new
compile), `--image/--last-image` (first/last-frame conditioning; 48xlarge only).

## 7. First run, warm runs and the compile cache

The first request on a new instance compiles the model's graphs for that canvas and frame count:

| | `trn2.48xlarge` | `trn2.3xlarge` |
|---|---:|---:|
| First request, nothing cached | ~12 min | ~45 min |
| First request, cache pulled from Hugging Face | — | ~30 min (graph loading; no compile) |
| Second request (one-time graph load) | normal | +~2–8 min |
| Later requests | warm times above | warm times above |

Run requests from one long-lived process (`--repeat`, several `--prompt`s, or your own loop around
`Omni.generate`); starting a new process reloads every graph.

**Reuse the compiled graphs on other instances** by pushing the caches to a Hugging Face repo once
and pulling them on each new host before its first run:

```bash
# on the host that compiled (needs a write token in HF_TOKEN):
sudo docker exec -e HF_TOKEN h3 python plugin/examples/minimax_h3/neuron_cache.py push <your-org>/minimax-h3-neuron-cache --private
# on a new host, after step 4:
sudo docker exec -e HF_TOKEN h3 python plugin/examples/minimax_h3/neuron_cache.py pull <your-org>/minimax-h3-neuron-cache
```

A pulled cache only applies to the same plugin commit, the same container image, the same
configuration (single device or 64 cores) and the same canvas, frame count and prompt-length bucket.
Anything else compiles as usual. Use `--subdir` to keep several combinations in one repo.

## Settings

`--single-device` applies these (any you set yourself take precedence):

| Variable | Value | What it does |
|---|---|---|
| `MINIMAX_H3_PER_BLOCK_MIN_ROWS` | `0` | the DiT runs as one graph per transformer block (the whole-model graph does not fit) |
| `MINIMAX_H3_ADALN_ON_HOST` | `1` | the 50 blocks' AdaLN projections are computed on the host (frees ~5 GB of device memory per core) |
| `MINIMAX_H3_HEADS_FIRST` | `bmm` | attention layout tuned for TP=4 |
| `MINIMAX_H3_FUSED_ADALN` | `1` | fuses the AdaLN modulation into the block graph |
| `MINIMAX_H3_ATTN_MASK_FIRST_TILE` | `1` | applies the padding mask on the first attention tile only |
| `MINIMAX_H3_EVICT_BLOCKS` | `10` | 10 DiT blocks leave device memory while the video decoder runs, from a copy kept on the host |
| `MINIMAX_H3_TE_STREAM` | `4` | the text encoder runs on the NeuronCores, its weights streamed from host memory 4 layers at a time |

Other variables:

| Variable | Default | What it does |
|---|---|---|
| `MINIMAX_H3_PDD_HEADS` | unset | path to `pdd_heads.safetensors`; required for PDD8 |
| `MINIMAX_H3_STAGE_PROFILE` | `0` | `1` logs each stage's time (encode, denoise, decode) per request |
| `MINIMAX_H3_HANDSHAKE_TIMEOUT_S` | `7200` | worker start-up timeout; raise it if a first-time compile is slower |

## Memory on a `trn2.3xlarge`

Device memory peaks at 22.6 GiB of 24 GiB per NeuronCore. On the host, the process needs more than
the instance's 124 GB: about 75 GB of weight copies (the streamed text encoder and the evicted DiT
blocks) plus the four workers, so part of it lives in swap. When the text encoder's host copy has
been swapped out, that request's prompt encoding takes 1-5 minutes instead of ~3.6 s; in our runs
this happened on some warm requests (130 s typical, 284 s worst). Do not run anything else large on
the same host, and run one generation process per `trn2.3xlarge`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Compilation error ... operation=model_default` or `[F137] neuronx-cc was forcibly killed` on the first request | the host ran out of memory during the compile | add swap (step 2); 192 GB on a `trn2.3xlarge` |
| `Allocation failure` / out of device memory on a `trn2.3xlarge` | a single-device setting was overridden, or a size other than 1344x768 / 124 frames | use `--single-device` without overrides; other sizes are not validated on one device |
| `PDD heads need num_inference_steps=9` | PDD8 run without `--steps 9` | add `--steps 9` |
| Every new process spends minutes before the first request | graphs load per process | keep one long-lived process; see section 7 |
| A video differs from one made with the text encoder on the host | expected: bf16 rounding in the text encoder changes the sample, not its quality | none needed |

## Limitations

- Text-to-video only on a single device; first/last-frame and reference modes need the
  `trn2.48xlarge`.
- One canvas size and frame count per compiled cache (1344x768, 124 frames as documented).
- Requires the specific container, plugin branch and `libtorch-neuronx-lite` build listed above.
