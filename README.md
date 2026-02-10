# Unsloth GPU Benchmark

**Plug-and-play GPU training benchmark** using [Unsloth](https://unsloth.ai) + synthetic data.  
Compare training performance across NVIDIA GPUs (RTX 5090, DGX Station GB300, RTX PRO 6000, etc.) with identical workloads — apples to apples.

## What It Does

- **Generates synthetic instruction-tuning data in-memory** — no dataset downloads needed
- **Fine-tunes a quantized LLM with QLoRA** via Unsloth (2x faster, 70% less VRAM)
- **Measures and logs**: training time, tokens/sec, VRAM peak, model load time
- **Saves JSON results** for side-by-side comparison across GPUs

## Profiles (Pick Based on Your GPU VRAM)

| Profile   | VRAM Needed | Model              | Seq Len | Batch | LoRA Rank |
|-----------|-------------|--------------------|---------|-------|-----------|
| `small`   | ~8 GB       | Llama 3.2 3B 4bit  | 2048    | 2     | 32        |
| `medium`  | ~16 GB      | Llama 3.1 8B 4bit  | 4096    | 2     | 64        |
| `large`   | ~24 GB      | Llama 3.1 8B 4bit  | 8192    | 4     | 128       |
| `xlarge`  | ~40 GB      | Qwen2.5 32B 4bit   | 8192    | 2     | 128       |
| `stress`  | ~60 GB      | Llama 3.1 70B 4bit | 4096    | 1     | 64        |
| `monster` | ~150 GB     | Llama 3.1 70B 4bit | 8192    | 4     | 128       |

### Recommended Profiles Per GPU

| GPU                          | VRAM    | Recommended Profile(s)           |
|------------------------------|---------|----------------------------------|
| **RTX 5090**                 | 32 GB   | `large`, `medium`                |
| **RTX PRO 6000 Blackwell**   | 48 GB   | `xlarge`, `large`                |
| **DGX Station GB300**        | 288 GB  | `monster`, `stress`, `all`       |

> **For apples-to-apples comparison**, run the **same profile** on all GPUs.  
> Use `xlarge` to compare all three (fits in 40 GB, every GPU can run it).  
> Use `monster` to push the DGX Station GB300 to its limits (70B model, batch=4, seq=8192).

---

## Quick Start with Docker (Recommended)

### Prerequisites
1. **Docker** installed ([Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows/Mac, or Docker Engine for Linux)
2. **NVIDIA Container Toolkit** installed ([guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
3. **NVIDIA GPU drivers** up to date

> **Important:** ARM64 Blackwell systems (DGX Station GB300) require a **custom Docker image** — they cannot use the standard x86 `unsloth/unsloth` image from Docker Hub. The `--dgx` flag handles this automatically.

### Option A: Headless / SSH-Only (Recommended)

No GUI, no Jupyter — just SSH in and run. Works on all machines.

**RTX 5090 (Windows PowerShell):**
```powershell
.\run_docker.ps1 -Mode run -Profile large
```

**RTX PRO 6000 (Linux / SSH):**
```bash
chmod +x run_docker.sh
./run_docker.sh --run xlarge
```

**DGX Station GB300 (Linux / SSH) — uses custom Dockerfile:**
```bash
chmod +x run_docker.sh
./run_docker.sh --run monster --dgx
```

The `--dgx` flag builds a custom image from `Dockerfile.dgx_station` (first run takes ~20-30 min to compile Triton + xFormers for ARM64/Blackwell). Subsequent runs reuse the cached image.

### Option B: Interactive (Jupyter Lab)

Only useful if you have a browser / GUI. Not needed for benchmarking.

```bash
./run_docker.sh --interactive large
# Open http://localhost:8888 (password: benchmark)
```

```powershell
.\run_docker.ps1 -Mode interactive -Profile large
```

### Option C: Manual Docker Run

```bash
# Create work directory and copy script
mkdir -p work && cp benchmark.py work/

# Standard GPUs (RTX 5090, RTX PRO 6000, etc.)
docker run --rm \
    --entrypoint bash \
    -v $(pwd)/work:/workspace/work \
    --gpus all \
    unsloth/unsloth \
    -c "cd /workspace/work && python benchmark.py --profile large"

# DGX Station GB300 (or any ARM64 Blackwell) — build custom image first
docker build -f Dockerfile.dgx_station -t unsloth-dgx .
docker run --rm \
    --entrypoint bash \
    -v $(pwd)/work:/workspace/work \
    --gpus all \
    unsloth-dgx \
    -c "cd /workspace/work && python benchmark.py --profile monster"
```

---

## Comparing Results

After running on each GPU, collect the JSON files from `work/` and run:

```bash
python compare_results.py work/benchmark_*.json
```

This prints a side-by-side table:

```
==========================================================================================
  PROFILE: LARGE  —  unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
==========================================================================================
Metric                         |   NVIDIA_RTX_5090    |   NVIDIA_RTX_PRO_6000
---------------------------------------------------------------------------
VRAM Total (GB)                |                32.0  |                 48.0
VRAM Peak (GB)                 |                24.1  |                 24.3
Training Time (s)              |               185.3  |                142.7
Tokens/sec                     |              12845   |               16632
Speedup vs RTX_5090            |               1.00x  |                1.30x
==========================================================================================
```

---

## Step-by-Step: Full 3-GPU Comparison

### 1. Test on Your RTX 5090 First (Windows)

```powershell
# Pull the image (one-time, ~15 GB)
docker pull unsloth/unsloth

# Run benchmark
cd c:\Users\Edwill\source\repos\unslothtraining
.\run_docker.ps1 -Mode run -Profile large

# Results saved to: work\benchmark_NVIDIA_GeForce_RTX_5090_large.json
```

### 2. Run on RTX PRO 6000 (SSH)

```bash
# From your local machine — copy files to the remote box
scp -r benchmark.py run_docker.sh Dockerfile.dgx_station user@rtx-pro-6000:/path/to/benchmark/

# SSH in
ssh user@rtx-pro-6000
cd /path/to/benchmark
chmod +x run_docker.sh
./run_docker.sh --run xlarge

# Copy results back to your local machine
# (from your local machine)
scp user@rtx-pro-6000:/path/to/benchmark/work/benchmark_*.json ./results/
```

### 3. Run on DGX Station GB300 (SSH Only)

```bash
# From your local machine — copy files to DGX Station
scp -r benchmark.py run_docker.sh Dockerfile.dgx_station user@dgx-station:/path/to/benchmark/

# SSH in
ssh user@dgx-station
cd /path/to/benchmark
chmod +x run_docker.sh

# The --dgx flag builds a custom image for ARM64/Blackwell (first run ~20-30 min)
./run_docker.sh --run monster --dgx

# Copy results back
# (from your local machine)
scp user@dgx-station:/path/to/benchmark/work/benchmark_*.json ./results/
```

### 4. Compare All Results

```bash
# Collect all JSON files into one folder, then:
python compare_results.py results/benchmark_*.json
```

---

## How It Works

1. **Synthetic Data**: Generates 1000–5000 instruction/response pairs in Alpaca format using templates and topic combinations. No internet needed after model download.

2. **Model Loading**: Downloads a 4-bit quantized model from Hugging Face (cached after first run).

3. **QLoRA Fine-Tuning**: Applies LoRA adapters to all linear layers and trains with Unsloth's optimized kernels. Uses gradient checkpointing and sequence packing.

4. **Metrics Collection**: Tracks wall-clock time, VRAM usage, throughput (tokens/sec, samples/sec), and training loss.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `docker: Error response from daemon: could not select device driver` | Install NVIDIA Container Toolkit |
| OOM (Out of Memory) | Use a smaller profile (`medium` or `small`) |
| Slow model download | First run downloads the model (~4-20 GB). Subsequent runs use cache. Mount a cache volume: `-v ~/.cache/huggingface:/home/unsloth/.cache/huggingface` |
| xFormers errors on Blackwell | The `unsloth/unsloth` image handles this. For ARM64 DGX systems, use `--dgx` flag (builds xFormers from source) |
| DGX: `exec format error` | You're using the x86 `unsloth/unsloth` image on ARM64. Use `--dgx` flag or build `Dockerfile.dgx_station` |
| DGX: Triton compile errors | The Dockerfile pins a known-good Triton commit. If issues persist, check CUDA 13.0 is installed: `nvcc --version` |

---

## Files

| File | Purpose |
|------|---------|
| `benchmark.py` | Main benchmark script with synthetic data generation |
| `run_docker.sh` | Linux/WSL/DGX Docker runner (supports `--dgx` flag for ARM64 Blackwell) |
| `run_docker.ps1` | Windows PowerShell Docker runner |
| `Dockerfile.dgx_station` | Custom Dockerfile for ARM64 Blackwell (DGX Station GB300) |
| `compare_results.py` | Side-by-side results comparison tool |
