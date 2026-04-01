# Solar-Open-100B Inference with Neuronx-Distributed

This directory contains the implementation and benchmark scripts for running inference with the **Solar-Open-100B Mixture of Experts (MoE)** model on AWS Trainium (Trn1) using `neuronx-distributed-inference` (NxDI).

## Overview

The Solar-Open-100B is a large-scale MoE model. This implementation provides a specialized Neuron architecture (`NeuronSolarOpenForCausalLM`) optimized for AWS Trainium, leveraging NxDI's Tensor Parallelism (TP) and Expert Parallelism (EP). 

Key features of this implementation include:
- **Expert Parallelism Support:** Efficient routing and computation distribution across NeuronCores using `moe_tp_degree` and `moe_ep_degree`.
- **Optimized MoE Routing:** Custom router implementation (`NeuronSolarOpenRouter`) handling Top-K routing.
- **Two-Phase Generation Support:** The benchmark script handles the model's distinct generation phases: outputting reasoning processes (Phase 1) followed by the final content (Phase 2).
- **Asynchronous Execution & On-Device Sampling:** Supports `async_mode` and `on_device_sampling_config` to minimize host-device communication bottlenecks and maximize throughput.

## File Structure

- `contrib/models/solar_open_100b_moe/src/modeling_solar_open_100b_moe.py`: The core model implementation. It defines the MoE layers, customized Attention (`NeuronSolarOpenAttention`), and the overall Causal LM class.
- `contrib/models/solar_open_100b_moe/test/solar_open_100b_moe_text_gen_benchmark.py`: An example script demonstrating how to compile, load, generate text, and benchmark the model's performance.

## Prerequisites

- **AWS EC2 Instance:** `trn1.32xlarge`.
- **AMI:** Deep Learning AMI Neuron (Ubuntu 24.04).
- **Installed packages:** `torch-neuronx 2.9.0`, `neuronx-distributed-inference 0.8.16251` or newer.
- **Model Weights:** Downloaded from Hugging Face (`upstage/Solar-Open-100B`).

## Usage Guide

### 1. Download Model Weights

Ensure you have the Hugging Face model weights available locally. The benchmark script defaults to `/home/ubuntu/workspace/model_hf/Solar-Open-100B`.

```bash
huggingface-cli download upstage/Solar-Open-100B --local-dir /home/ubuntu/workspace/model_hf/Solar-Open-100B
```

### 2. Configure Compilation

The benchmark script (`solar_open_100b_moe_text_gen_benchmark.py`) demonstrates the recommended compiler flags and configuration.

**Key Configuration Highlights:**
* `tp_degree=32`
* `moe_tp_degree=8`
* `moe_ep_degree=4`
* `batch_size=16`
* `async_mode=True`

> **Note:** The script includes specific environment variables (`NEURON_CC_FLAGS`) optimized for this model.

### 3. Run Inference and Benchmark

Execute the benchmark script. If the model is not yet compiled, it will automatically compile it and save the artifacts before running the generation.

```bash
# Run full model benchmark
python test/solar_open_100b_moe_text_gen_benchmark.py

# Run a smaller layer test (e.g., 2 layers) for quick validation
python test/solar_open_100b_moe_text_gen_benchmark.py --num_layers 2
```

### Output Interpretation

The script performs two main actions:

* **Text Generation:** It runs a two-phase prompt ("Could you explain the concept of quantum computing in simple terms?"). It will print "Phase 1: Reasoning Process" and "Phase 2: Final Content" to demonstrate the model's output.
* **Benchmarking:** It executes `benchmark_sampling` (20 runs by default) to measure Throughput (tokens/s) and Latency (ms). The results are saved to a JSON file (e.g., `solar_open_100b_benchmark_report_full.json`).