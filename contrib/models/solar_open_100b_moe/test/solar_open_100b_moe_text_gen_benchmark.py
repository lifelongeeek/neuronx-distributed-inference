import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, GenerationConfig

os.environ["NEURON_CC_FLAGS"] = (
    "--cache_dir=/var/tmp/compiler_cache "
    "--enable-saturate-infinity "
    "--model-type transformer -O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--auto-cast=none "
    "--internal-enable-dge-levels vector_dynamic_offsets "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from modeling_solar_open_100b_moe import (
    SolarOpenInferenceConfig, 
    NeuronSolarOpenForCausalLM, 
    load_solar_open_config
)

from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

MODEL_PATH = "/home/ubuntu/workspace/model_hf/Solar-Open-100B"

def main():
    parser = argparse.ArgumentParser(description="Solar-Open-100B NeuronX Inference")
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=None, 
        help="Number of layers to compile and run (uses all layers if not provided)"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Batch size for compilation and inference"
    )
    args = parser.parse_args()

    if args.num_layers is not None:
        traced_model_path = f"/home/ubuntu/workspace/neuronx-distributed-inference/traced_solar_open_100b_{args.num_layers}layer"
        layer_info_str = f"{args.num_layers} Layers (EP=4, TP=8, BS={args.batch_size})"
    else:
        traced_model_path = f"/home/ubuntu/workspace/neuronx-distributed-inference/traced_solar_open_100b_full_layer"
        layer_info_str = f"Full-Layer (EP=4, TP=8, BS={args.batch_size})"

    print(f"Initializing MoE Neuron Configuration for Trn1 (Generation Mode, {layer_info_str})...")
    
    neuron_config = MoENeuronConfig(
        tp_degree=32,   
        moe_tp_degree=8,                
        moe_ep_degree=4,                
        cp_degree=1,                   
        attention_dp_degree=1,         
        logical_nc_config=1,           
        batch_size=args.batch_size,
        max_context_length=1024,       
        seq_len=2048,
        torch_dtype=torch.bfloat16,    
        
        fused_qkv=False,
        qkv_kernel_enabled=False, 
        sequence_parallel_enabled=False,
        shared_experts_sequence_parallel_enabled=False, 
        use_index_calc_kernel=False,
        moe_mask_padded_tokens=True,
        
        blockwise_matmul_config={"use_shard_on_intermediate_dynamic_while": False, "skip_dma_token": True},
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        ),
        async_mode=True, 
        padding_side="right"
    )

    inference_config = SolarOpenInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_solar_open_config(MODEL_PATH)
    )
    
    if args.num_layers is not None:
        inference_config.num_hidden_layers = args.num_layers

    model = NeuronSolarOpenForCausalLM(MODEL_PATH, inference_config)
    
    if os.path.exists(traced_model_path):
        print(f"Loading compiled traced model from {traced_model_path}...")
        model.load(traced_model_path)
    else:
        print(f"Traced model not found. Compiling the model to {traced_model_path}...")
        model.compile(traced_model_path)
        print(f"Loading the newly compiled model...")
        model.load(traced_model_path)

    print("Loading Tokenizer...")
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "Could you explain the concept of quantum computing in simple terms?"
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        [text] * args.batch_size, 
        return_tensors="pt",
        padding=True
    )
    
    input_length = inputs["input_ids"].shape[1]
    
    generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    generation_config.max_new_tokens = 1024
    generation_config.pad_token_id = tokenizer.pad_token_id
    
    stop_tokens = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "additional_special_tokens_ids") and tokenizer.additional_special_tokens_ids:
        stop_tokens.extend(tokenizer.additional_special_tokens_ids)
    for st in ["<|end|>", "<|im_end|>", "<|eot_id|>"]:
        tid = tokenizer.convert_tokens_to_ids(st)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in stop_tokens:
            stop_tokens.append(tid)

    generation_config.eos_token_id = stop_tokens
    generation_model = HuggingFaceGenerationAdapter(model)

    # ---------------------------------------------------------
    # [Phase 1] Thinking Process
    # ---------------------------------------------------------
    print(f"\nGenerating outputs on Trainium Hardware... (Phase 1: Thinking, Max Tokens: {generation_config.max_new_tokens})")
    with torch.no_grad():
        think_outputs = generation_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config
        )
    
    think_generated_tokens = think_outputs[0, input_length:]
    
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    end_positions = (think_generated_tokens == end_token_id).nonzero(as_tuple=True)[0]
    
    if len(end_positions) > 0:
        end_idx = end_positions[0].item()
        valid_think_tokens = think_generated_tokens[:end_idx + 1]
        
        valid_think_outputs = think_outputs[:, :input_length + end_idx + 1]
        
        think_text = tokenizer.decode(valid_think_tokens, skip_special_tokens=False)
        print("\n================== Phase 1: Reasoning Process ==================")
        print(think_text)
        print("==========================================================================\n")

        # ---------------------------------------------------------
        # [Phase 2] Final Content
        # ---------------------------------------------------------
        print(f"Generating outputs on Trainium Hardware... (Phase 2: Content)")
        
        content_tag = "<|begin|>assistant<|content|>"
        tag_tokens = tokenizer(content_tag, return_tensors="pt", add_special_tokens=False)["input_ids"]
        tag_tokens_batch = tag_tokens.repeat(args.batch_size, 1)
        
        new_input_ids = torch.cat([valid_think_outputs, tag_tokens_batch], dim=-1)
        new_attention_mask = torch.ones_like(new_input_ids)
        
        current_length = new_input_ids.shape[1]
        remaining_space = neuron_config.seq_len - current_length
        
        if remaining_space <= 0:
            print("Error: Input length already exceeds seq_len. Cannot generate phase 2.")
        else:
            generation_config.max_new_tokens = min(1024, remaining_space)
            
            with torch.no_grad():
                final_outputs = generation_model.generate(
                    new_input_ids,
                    attention_mask=new_attention_mask,
                    generation_config=generation_config
                )
                
            content_start_idx = new_input_ids.shape[1]
            final_generated_tokens = final_outputs[0, content_start_idx:]
            
            final_generated_text = tokenizer.decode(final_generated_tokens, skip_special_tokens=True)
            
            print("\n================== Phase 2: Final Content ==================")
            print(final_generated_text)
            print("======================================================================\n")

    else:
        think_text = tokenizer.decode(think_generated_tokens, skip_special_tokens=False)
        print("\n================== Phase 1: Reasoning Process (Incomplete) ==================")
        print(think_text)
        print("\nWarning: The model did not terminate normally with the <|end|> token (e.g., reached max_new_tokens)")

    print("\n====================== Starting Performance Benchmark ======================")
    report_path = f"solar_open_100b_benchmark_report_{args.num_layers if args.num_layers else 'full'}.json"

    generation_config.max_new_tokens = 1024

    report = benchmark_sampling(
        model=model,
        generation_config=generation_config,
        target="all",
        num_runs=20,
        benchmark_report_path=report_path
    )

    print("\nBenchmark successfully completed!")
    print(f"Check '{report_path}' for the detailed report.")    

if __name__ == "__main__":
    main()
