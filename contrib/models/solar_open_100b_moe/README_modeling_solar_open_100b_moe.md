# Solar-Open-100B Inference with Neuronx-Distributed

This directory contains the implementation and benchmark scripts for running inference with the **Solar-Open-100B Mixture of Experts (MoE)** model on AWS Trainium (Trn1) using `neuronx-distributed-inference` (NxDI).

## Overview

The Solar-Open-100B is a large-scale MoE model (128 routed experts + 1 shared expert, top_k=6). This implementation provides a specialized Neuron architecture (`NeuronSolarOpenForCausalLM`) optimized for AWS Trainium, leveraging NxDI's Tensor Parallelism (TP) and Expert Parallelism (EP).

Key features:
- **Expert Parallelism:** Efficient routing and computation distribution using `moe_tp_degree=8`, `moe_ep_degree=4`.
- **FP8 Quantization:** Per-channel symmetric FP8 (E4M3) weight quantization with EP-aware scale requantization for accuracy preservation.
- **Two-Phase Generation:** Handles the model's Thinking (Phase 1) and Content (Phase 2) generation modes.
- **On-Device Sampling:** Supports `on_device_sampling_config` to minimize host-device communication.

## File Structure

```
contrib/models/solar_open_100b_moe/
├── src/
│   └── modeling_solar_open_100b_moe.py          # Core model implementation
└── test/
    ├── quantize_solar_100b.py                    # FP8 quantization script
    └── solar_open_100b_moe_text_gen_benchmark.py # Inference & benchmark script
```

### File Descriptions

| File | Description |
|------|-------------|
| `src/modeling_solar_open_100b_moe.py` | MoE model definition. Includes `NeuronSolarOpenForCausalLM`, custom router (`NeuronSolarOpenRouter`), attention, state dict conversion (`convert_solar_open_hf_to_neuron_state_dict`), and expert weight merging with FP8 scale requantization (`_merge_expert_weights`). |
| `test/quantize_solar_100b.py` | HuggingFace model weights to FP8 (E4M3) per-channel symmetric quantization. Supports `--num-layers` for partial layer quantization (fast testing). |
| `test/solar_open_100b_moe_text_gen_benchmark.py` | Model compilation, text generation (two-phase), and performance benchmarking. Supports both BF16 and FP8 quantized inference via `--quantized` flag. |

## Prerequisites

- **Instance:** `trn1.32xlarge` (32 NeuronCores)
- **AMI:** Deep Learning AMI Neuron (Ubuntu 24.04)
- **Packages:** `torch-neuronx 2.9.0`, `neuronx-distributed-inference 0.8.16251` or newer
- **Model Weights:** `upstage/Solar-Open-100B` from Hugging Face

## Quick Start

### 1. Download Model Weights

```bash
huggingface-cli download upstage/Solar-Open-100B \
  --local-dir /home/ubuntu/workspace/model_hf/Solar-Open-100B
```

### 2. Run BF16 Inference (No Quantization)

```bash
cd /home/ubuntu/workspace/neuronx-distributed-inference/contrib/models/solar_open_100b_moe

# Full model
python test/solar_open_100b_moe_text_gen_benchmark.py

# Quick 2-layer validation
python test/solar_open_100b_moe_text_gen_benchmark.py --num_layers 2
```

### 3. Run FP8 Quantized Inference

```bash
# Step 1: Generate quantized checkpoint
python test/quantize_solar_100b.py

# Step 2: Run quantized inference
python test/solar_open_100b_moe_text_gen_benchmark.py --quantized
```

## Detailed Usage

### quantize_solar_100b.py

Generates FP8 (E4M3) per-channel symmetric quantized checkpoints from HuggingFace model weights.

```bash
# Full 48-layer quantization
python test/quantize_solar_100b.py

# 2-layer quantization (fast testing)
python test/quantize_solar_100b.py --num-layers 2

# Custom paths
python test/quantize_solar_100b.py \
  --model-path /path/to/Solar-Open-100B \
  --output-path /path/to/quantized-output \
  --num-layers 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `/home/ubuntu/workspace/model_hf/Solar-Open-100B` | HuggingFace model directory |
| `--output-path` | Auto-generated | Quantized checkpoint save path |
| `--num-layers` | All (48) | Number of layers to quantize |

**Output path convention (auto-generated):**
- Full layer: `Solar-Open-100B-f8e4m3`
- N layers: `Solar-Open-100B-f8e4m3-{N}layer`

**Quantization details:**
- **dtype:** `float8_e4m3fn` (FP8 E4M3)
- **method:** `per_channel_symmetric`
- **modules excluded:** `lm_head`, `mlp.gate` (router), `o_proj`, `shared_experts`
- **Expert scale handling:** EP max scale reduction with FP8 weight requantization to preserve per-expert accuracy

### solar_open_100b_moe_text_gen_benchmark.py

Compiles, loads, runs text generation, and benchmarks the model.

```bash
# BF16 full model
python test/solar_open_100b_moe_text_gen_benchmark.py

# BF16 2-layer test
python test/solar_open_100b_moe_text_gen_benchmark.py --num_layers 2

# FP8 quantized full model
python test/solar_open_100b_moe_text_gen_benchmark.py --quantized

# FP8 quantized 2-layer test
python test/solar_open_100b_moe_text_gen_benchmark.py --num_layers 2 --quantized

# Custom batch size and quantized checkpoint path
python test/solar_open_100b_moe_text_gen_benchmark.py \
  --quantized \
  --batch_size 8 \
  --quantized_checkpoints_path /path/to/quantized-checkpoint
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_layers` | All (48) | Number of layers to compile and run |
| `--batch_size` | 16 | Batch size for inference |
| `--quantized` | False | Enable FP8 quantized inference |
| `--quantized_checkpoints_path` | Auto-resolved | Path to quantized checkpoints |

**Model configuration:**
- `tp_degree=32`, `moe_tp_degree=8`, `moe_ep_degree=4`
- `max_context_length=1024`, `seq_len=2048`
- `on_device_sampling`: top_k=50, top_p=0.95, temperature=0.8

**When `--quantized` is specified:**
- Environment variables `XLA_HANDLE_SPECIAL_SCALAR=1` and `UNSAFE_FP8FNCAST=1` are set automatically
- Quantized checkpoint path is auto-resolved based on `--num_layers`
- Compiled model is saved separately with `_quant` suffix to avoid conflicts

### modeling_solar_open_100b_moe.py

Core model implementation. Not executed directly, but imported by the scripts above.

**Key classes:**

| Class | Description |
|-------|-------------|
| `NeuronSolarOpenForCausalLM` | Main Causal LM class. Handles HF model loading, state dict conversion, and compilation. |
| `NeuronSolarOpenModel` | Decoder layer stack with embedding and lm_head. |
| `NeuronSolarOpenDecoderLayer` | Single decoder layer: Attention + MoE. |
| `NeuronSolarOpenAttention` | GQA attention with optional QK norm. |
| `NeuronSolarOpenRouter` | Custom MoE router with group-limited top-k routing and sigmoid scoring. |
| `SolarOpenInferenceConfig` | Model configuration with MoE-specific parameter handling. |

**Key functions:**

| Function | Description |
|----------|-------------|
| `convert_solar_open_hf_to_neuron_state_dict()` | Converts HF checkpoint to Neuron format. Handles attention key renaming, expert weight merging, shared expert padding, and FP8 scale processing. |
| `_merge_expert_weights()` | Merges per-expert weights into `ExpertMLPsV2` format. For quantized models, performs EP max scale reduction with FP8 weight requantization to avoid 3-7x overscaling. |
| `load_solar_open_config()` | Loads model config from `config.json` with Solar-specific defaults. |

## Quantization Architecture

### FP8 Weight Quantization Flow

```
HuggingFace Model (BF16)
  │
  ├── quantize_pytorch_model_per_channel_symmetric()
  │     → FP8 weights + per-channel scales
  │
  ├── Modules NOT quantized (kept BF16):
  │     lm_head, mlp.gate (router), o_proj, shared_experts
  │
  └── Save as safetensors
        → Solar-Open-100B-f8e4m3/model.safetensors

Inference Load (convert_solar_open_hf_to_neuron_state_dict)
  │
  ├── Attention Q/K/V: FP8 + scale → QuantizedColumnParallel
  ├── Attention O: BF16 → RowParallelLinear
  ├── Expert weights: FP8 + per-expert scale
  │     → EP max scale reduction + weight requantization
  │     → [E, H, 2*I] weight + [ep, 1, 2*I] scale
  └── Shared experts: BF16 → ColumnParallel/RowParallel
```

### Expert Scale Requantization

Standard EP max scale reduction causes 3-7x overscaling because one outlier expert inflates the scale for all 32 experts in the same EP group. This implementation requantizes FP8 values to match the group max scale:

```
For each EP group:
  1. max_scale = max(expert_scales within group)
  2. For each expert:
     float_w = fp8_w * expert_scale          (dequantize)
     new_fp8_w = clamp(float_w / max_scale)  (requantize with group scale)
  3. Store new_fp8_w + max_scale
```

Result: `new_fp8_w * max_scale ≈ original_weight` (cosine similarity > 0.999)

## Known Limitations

- **Blockwise kernel FP8:** The current NxD blockwise matmul kernel does not support FP8 native compute. FP8 expert weights are dequantized to BF16 before kernel execution, which adds runtime overhead. On Trn2, MXFP4 (Microscaling FP4) is available as a native compute alternative.
- **QKV kernel incompatibility:** `qkv_kernel_enabled=True` is not compatible with FP8 quantized weights due to weight transpose handling in `QuantizedColumnParallel`. Set to `False` when using quantization.
