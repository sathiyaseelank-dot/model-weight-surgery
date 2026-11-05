# Remote Host Devices & Pipeline Parallelism

## Remote Host Devices (SSH-accessible LAN hosts)

| # | IP | Hostname | SSH User | RAM (GiB) | Cores | GPU Hardware |
|---|----|----------|----------|-----------|-------|--------------|
| 1 | 192.168.1.97 | inkyank-08 | racer | 7.57 | 6 | Intel UHD Graphics 630 (iGPU) |
| 2 | 192.168.1.38 | luffy | nika | 15.36 | 12 | Intel UHD Graphics 730 (iGPU) |
| 3 | 192.168.1.75 | archlinux | inkyank | 7.57 | 6 | Intel UHD Graphics 630 (iGPU) |
| 4 | 192.168.1.42 | archlinux | seechan5 | 15.36 | 12 | Intel UHD Graphics 730 (iGPU) |
| 5 | 192.168.1.79 | archlinux | seechanlabs | 15.36 | 12 | Intel UHD Graphics 730 (iGPU) |
| 6 | 192.168.1.53 | omarchy | manojinkyank | 15.35 | 12 | Intel UHD Graphics 730 (iGPU) |
| 7 | 192.168.1.65 | kishor | vegeta | 15.52 | 8 | **NVIDIA GeForce GTX 960M (2 GB VRAM)** + Intel HD 530 |
| 8 | 192.168.1.56 | archlinux | seechan3 | 15.36 | 12 | Intel UHD Graphics 730 (iGPU) |
| 9 | 192.168.1.43 | (unknown) | (blocked) | — | — | (Rate-limited) |
| - | 192.168.1.164 | inkyank-01 | (controller) | 15.36 | 12 | Intel UHD Graphics 730 (iGPU) |

### GPU Discovery Details
- **Dedicated GPU Host:** **`192.168.1.65`** (`vegeta@kishor`)
  - **GPU Model:** NVIDIA GeForce GTX 960M (GM107M, Maxwell)
  - **Dedicated VRAM:** 2048 MiB (2.0 GB GDDR5)
  - **NVIDIA Driver:** `580.178.04`
  - **CUDA Version:** `13.0`
  - **Current GPU Status:** 0 MiB in use, 100% free VRAM, temperature 45°C.
- All other compute hosts rely solely on integrated Intel UHD/HD graphics.

### Total Available Resources (8 queried hosts)

- **Total RAM:** 112,654,664 kB (≈ 107.4 GiB)
- **Total Cores:** 80

### Local Controller (not counted above)

- 192.168.1.164 (inkyank-01-computer): 16,106,512 kB (≈ 15.4 GiB), 12 cores

### Notes

- Host 192.168.1.43 (9th remote) is a Linux host (SSH on port 22) but currently rate-limited ("Not allowed at this time"), so its RAM/cores could not be queried.
- Host 192.168.1.36 is a camera/IoT device (HTTP/RTSP), not a compute host.
- Host 192.168.1.56 was accessed with an empty password.

---

## Pipeline Parallelism — llama.cpp RPC (Distributed LLM Inference)

### Session Overview

- **Conversation ID:** 8f73e441-1c99-4799-b007-3df37ce41d7d
- **Project:** /home/inkyank/orca/projects/distibuted computing
- **Dates:** 2026-10-24 → 2026-10-29
- **Topic:** Running llama3.2:3b distributed across two LAN machines using llama.cpp RPC pipeline parallelism

### Machines

- **Main/driver:** 192.168.1.75 (i5-8500, 8 GB, Arch Linux)
- **Worker:** 192.168.1.42 (i5-12400, 16 GB, Arch Linux) running ggml-rpc-server on port 50052

### Build

- Built llama.cpp with `-DGGML_RPC=ON` on both machines.

### Model Setup

- Used the Ollama-cached GGUF blob (sha256-dde5aa...) and created a `llama-3.2-3b.gguf` symlink.

### KV-Cache Fix

- Fixed a huge KV-cache allocation by setting `-c 2048`.

### Parallelism Tests & Benchmarks

#### Historical Benchmarks (from session log)
- **Pipeline parallelism:** `-sm layer` → pp32 55.73 t/s, tg16 7.12 t/s
- **Tensor parallelism:** `-sm tensor` → pp32 46.47 t/s, tg16 7.39 t/s

#### Live LAN Benchmark Results (2026-11-02 Run: Driver 192.168.1.75 + Worker 192.168.1.42:50052)
- **Model:** Llama 3.2 3B (Q4_K_M, 1.87 GiB, 3.21B params)
- **Configuration:** 14 layers offloaded to `RPC0` (`192.168.1.42:50052`), 4 threads (`-t 4`), context 2048 (`-c 2048`).

| Parallelism Mode | Benchmark / Test | Flag | Prompt Processing (`pp32`) | Text Generation (`tg16`) |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline Parallelism** | `llama-bench` | `-sm layer` | **59.57 t/s** | **11.22 t/s** |
| **Pipeline Parallelism** | `llama-cli` | `-sm layer` | **71.80 t/s** | **10.80 t/s** |
| **Tensor Parallelism** | `llama-bench` | `-sm tensor` | **70.15 t/s** | **10.09 t/s** |
| **Tensor Parallelism** | `llama-cli` | `-sm tensor` | **64.80 t/s** | **10.40 t/s** |

---

## GPU Hardware Acceleration — GTX 960M Vulkan RPC Worker

### Hardware & Backend Architecture
- **Host:** `192.168.1.65` (`vegeta@kishor`)
- **GPU:** NVIDIA GeForce GTX 960M (GM107M, Maxwell architecture, 2048 MiB GDDR5 VRAM).
- **Driver:** NVIDIA Proprietary `580.178.04`, Vulkan API `1.4.312`.
- **Backend Strategy:** Because Maxwell (`sm_50`) was deprecated in modern CUDA releases (CUDA 13+), `llama.cpp`'s `ggml-rpc-server` was compiled on `.65` with native Vulkan acceleration (`-DGGML_VULKAN=ON -DGGML_RPC=ON`).
- **Exposed Network Devices:**
  - `RPC0`: Intel(R) HD Graphics 530 (11,917 MiB total shared memory)
  - `RPC1` (or `RPC6` in full cluster): **NVIDIA GeForce GTX 960M (2,294 MiB total, 2,232 MiB free VRAM)**

### Layer Allocation & Memory Budget
The Llama 3.2 3B Q4_K_M model consists of 28 transformer blocks:
- **Total Model Weights:** ~2,068 MiB
- **Layer Allocation:** 24 layers offloaded to GPU VRAM (~1,772 MiB weights + ~216 MiB KV cache + compute buffers), fitting within the 2,232 MiB free VRAM headroom.
- Remaining 4 layers are processed on CPU host.

### Live Benchmark Results & Speedup Comparison

| Mode / Hardware | Offload Configuration | Prompt Processing (`pp32`) | Text Generation (`tg16`) | Speedup vs CPU |
| :--- | :--- | :--- | :--- | :--- |
| **Pure CPU Distributed (8 Hosts)** | 28 layers distributed across 8 CPUs (`-sm layer`, `ts=0.00`) | **46.74 ± 1.61 t/s** | **6.81 ± 0.40 t/s** | 1.00x (Baseline) |
| **Pure CPU Single Worker (.42)** | 14 layers offloaded to .42 CPU (`-sm layer`) | **59.57 t/s** | **11.22 t/s** | 1.65x |
| **Strategy A: GPU-Heavy, Small on CPUs** | **GTX 960M GPU gets ~18–20 layers, 1 layer on each CPU host (`-ts 1,1,1,1,1,1,0,18`)** | **52.65 ± 1.97 t/s** | **11.78 ± 0.25 t/s** | **1.73x (+73.0%)** |
| **Strategy B: Equal Split Across Hosts** | **Equal distribution on all CPU hosts + GPU (`-ts 1,1,1,1,1,1,0,1`)** | **52.82 ± 0.23 t/s** | **11.98 ± 0.09 t/s** | **1.76x (+75.9%)** |
| **Strategy C: Core-Weighted / Efficiency-Aware** | **12-core hosts (.38, .42, .79, .56) get 4x layers, 8-core gets 2x, 6-core gets 1x, GPU gets 8x (`-ts 1,4,4,4,2,4,0,8`)** | **53.37 ± 0.91 t/s** | **11.75 ± 0.18 t/s** | **1.73x (+72.5%)** |
| **GTX 960M GPU RPC Worker (.65)** | **24 layers offloaded to GTX 960M Vulkan** (`--device RPC1 -ngl 24`) | **70.20 t/s** | **17.10 t/s** | **2.41x (Fastest)** |
| **GTX 960M GPU RPC Worker (.65)** | 20 layers offloaded to GTX 960M Vulkan (`--device RPC1 -ngl 20`) | **36.20 t/s** | **16.60 t/s** | 2.34x |

#### Host Binary Status & Cluster Deployment
- `192.168.1.75` (Driver): `llama.cpp/build/bin/` (`llama-cli`, `llama-bench`, `ggml-rpc-server`, model file `llama-3.2-3b.gguf` present).
- `192.168.1.42` (Worker): `~/llama.cpp/build/bin/` (`ggml-rpc-server` running on `0.0.0.0:50052` with `-t 10`).
- Remaining 6 hosts (`.97`, `.38`, `.79`, `.53`, `.65`, `.56`): Packaged `ggml-rpc-server` and required `libggml*` runtime libraries into `rpc-bundle.tar.gz`, deployed to `~/llama-rpc` on all hosts, and launched with per-host CPU thread allocations without requiring sudo.

#### Full 8-Host Cluster Devices (102,263 MiB Free Worker RAM Registered)
- `RPC0`: `192.168.1.97:50052` (7,751 MiB free, threads: 4)
- `RPC1`: `192.168.1.38:50052` (15,725 MiB free, threads: 10)
- `RPC2`: `192.168.1.42:50052` (15,726 MiB free, threads: 10)
- `RPC3`: `192.168.1.79:50052` (15,725 MiB free, threads: 10)
- `RPC4`: `192.168.1.53:50052` (15,720 MiB free, threads: 10)
- `RPC5`: `192.168.1.65:50052` (15,889 MiB free, threads: 6)
- `RPC6`: `192.168.1.56:50052` (15,727 MiB free, threads: 10)
- Driver: `192.168.1.75` (coordinator / prompt orchestration, threads: 4)

#### Verified Per-Host Layer & Compute Breakdown (from llama.cpp runtime log)

Every single one of the 8 hosts actively holds model layers, KV-cache, and active forward-pass compute buffers:

| Host / Node | Role | Model Weights (MiB) | Context / KV Cache (MiB) | Active Compute Buffer (MiB) | Total Active RAM (MiB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **192.168.1.75** | Host / Driver | 330.00 MiB | 8.00 MiB | 80.02 MiB | 418.02 MiB |
| **192.168.1.97** | RPC0 Worker | 169.80 MiB | 24.00 MiB | 62.01 MiB | 255.81 MiB |
| **192.168.1.38** | RPC1 Worker | 229.24 MiB | 32.00 MiB | 62.01 MiB | 323.25 MiB |
| **192.168.1.42** | RPC2 Worker | 223.83 MiB | 32.00 MiB | 62.01 MiB | 317.84 MiB |
| **192.168.1.79** | RPC3 Worker | 284.04 MiB | 40.00 MiB | 62.01 MiB | 386.05 MiB |
| **192.168.1.53** | RPC4 Worker | 229.24 MiB | 32.00 MiB | 62.01 MiB | 323.25 MiB |
| **192.168.1.65** | RPC5 Worker | 230.02 MiB | 32.00 MiB | 62.01 MiB | 324.03 MiB |
| **192.168.1.56** | RPC6 Worker | 491.20 MiB | 24.00 MiB | 62.01 MiB | 577.21 MiB |
| **Total** | **8 Hosts** | **2,187.37 MiB** | **224.00 MiB** | **514.09 MiB** | **2,925.46 MiB** |

During every generated token in pipeline parallelism (`-sm layer`), each host computes its assigned transformer layer blocks sequentially and streams the intermediate tensor activations to the next node across the LAN.

| Distributed Mode | Test / Command | Flag | Prompt Processing (`pp32`) | Text Generation (`tg16`) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full 8-Host Pipeline Parallelism** | `llama-bench` | `-sm layer` | **50.71 t/s** | **7.10 t/s** | Activations passed stage-by-stage across LAN nodes |
| **Full 8-Host Pipeline Parallelism** | `llama-cli` | `-sm layer` | **42.80 t/s** | **6.70 t/s** | Full interactive prompt generation across all 8 nodes |
| **Full 8-Host Tensor Parallelism** | `llama-bench` | `-sm tensor` | **5.49 t/s** | **0.74 t/s** | Weight/KV matrices sharded across all 7 RPC sockets (heavy network synchronization overhead per token) |

### Comparisons

- Compared against Ollama (which does not support multi-node distributed inference).
- Discussed other distributed options: Petals, vLLM+Ray, etc.

### Where It's Stored

- Conversation DB: `~/.gemini/antigravity-cli/conversations/8f73e441-1c99-4799-b007-3df37ce41d7d.db`
- Brain/logs: `~/.gemini/antigravity-cli/brain/8f73e441-1c99-4799-b007-3df37ce41d7d/`
- Note: Another conversation DB (d5744f2a-...) also matched the search terms, but that session is about the agent-approval-watcher skill, not distributed LLM inference.

---

## Local Model & Resource Allocation

### Local Model (from conversation)

- **Model:** Llama 3.2 3B (llama3.2:3b / Llama-3.2-3B-Instruct, GGUF Q4_K_M, ~2 GB)
- **Runtime:** llama.cpp (with RPC for distributed inference)
- **Other models referenced:** Llama 3.1 8B, DeepSeek-R1 14B

### Allocation Plan (leave headroom for regular host use)

Reserve ~4 GiB RAM and 2 cores per host for the OS / regular user; the remainder is available for the local model run.

| Host (IP) | Total RAM (GiB) | Regular Use (GiB) | Model RAM (GiB) | Total Cores | Regular Use Cores | Model Cores |
|-----------|-----------------|-------------------|-----------------|-------------|-------------------|-------------|
| 192.168.1.97 | 7.57 | 3.00 | 4.57 | 6 | 2 | 4 |
| 192.168.1.38 | 15.36 | 4.00 | 11.36 | 12 | 2 | 10 |
| 192.168.1.75 | 7.57 | 3.00 | 4.57 | 6 | 2 | 4 |
| 192.168.1.42 | 15.36 | 4.00 | 11.36 | 12 | 2 | 10 |
| 192.168.1.79 | 15.36 | 4.00 | 11.36 | 12 | 2 | 10 |
| 192.168.1.53 | 15.35 | 4.00 | 11.35 | 12 | 2 | 10 |
| 192.168.1.65 | 15.52 | 4.00 | 11.52 | 8 | 2 | 6 |
| 192.168.1.56 | 15.36 | 4.00 | 11.36 | 12 | 2 | 10 |

### Pooled Model Resources (8 hosts)

- **Model RAM pool:** ≈ 77.45 GiB
- **Model core pool:** 64 cores
