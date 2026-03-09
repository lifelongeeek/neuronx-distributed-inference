import os
import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.solar_open.modeling_solar_open_baseline_v0 import (
    SolarOpenInferenceConfig, 
    NeuronSolarOpenForCausalLM, 
    load_solar_open_config
)

from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

MODEL_PATH = "/home/ubuntu/workspace/model_hf/Solar-Open-100B"
TRACED_MODEL_PATH = "/home/ubuntu/workspace/neuronx-distributed-inference/traced_solar_open_100b_full_layer" 

def main():
    rank = int(os.environ.get("RANK", "0"))
    
    print("Initializing MoE Neuron Configuration for Trn1 (Generation Mode)...")
    
    os.environ["NEURON_CC_FLAGS"] = (
        "--cache_dir=/home/ubuntu/workspace/compiler_cache "
        "--enable-saturate-infinity "
        "--model-type transformer -O1 "
        "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
        "--auto-cast=none "
        "--internal-enable-dge-levels vector_dynamic_offsets "
        "--internal-hlo2tensorizer-options='--verify-hlo=true'"
    )

    neuron_config = MoENeuronConfig(
        tp_degree=32,                  
        moe_ep_degree=1,                
        moe_tp_degree=32,              
        cp_degree=1,                   
        attention_dp_degree=1,         
        logical_nc_config=1,           
        batch_size=1,
        max_context_length=128,       
        seq_len=512,
        torch_dtype=torch.bfloat16,    
        
        fused_qkv=False,
        qkv_kernel_enabled=False, 
        sequence_parallel_enabled=False,
        shared_experts_sequence_parallel_enabled=False, 
        use_index_calc_kernel=False,
        moe_mask_padded_tokens=True,
        
        blockwise_matmul_config={"use_shard_on_intermediate_dynamic_while": False, "skip_dma_token": True},
        on_device_sampling_config=None,
        async_mode=False, 
        padding_side="right"
    )

    inference_config = SolarOpenInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_solar_open_config(MODEL_PATH)
    )

    model = NeuronSolarOpenForCausalLM(MODEL_PATH, inference_config)
    
    if os.path.exists(TRACED_MODEL_PATH):
        print(f"Loading compiled traced model from {TRACED_MODEL_PATH}...")
        model.load(TRACED_MODEL_PATH)
    else:
        print(f"Traced model not found. Compiling the model to {TRACED_MODEL_PATH}...")
        model.compile(TRACED_MODEL_PATH)
        print(f"Loading the newly compiled model...")
        model.load(TRACED_MODEL_PATH)

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
        text, 
        return_tensors="pt"
    )
    
    input_length = inputs["input_ids"].shape[1]
    print(f"\nGenerating outputs on Trainium Hardware... (Greedy Search, Max New Tokens: 384)")
    
    generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 1
    generation_config.max_new_tokens = 384  
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

    with torch.no_grad():
        outputs = generation_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config
        )
    
    generated_tokens = outputs[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print("\n================== NxDI Hardware Generation Result (100B Full-Layer) ==================")
    print(generated_text)
    print("=======================================================================================\n")
    
    torch.save(generated_tokens.cpu(), "nxdi_generated_tokens_full.pt")

if __name__ == "__main__":
    main()
