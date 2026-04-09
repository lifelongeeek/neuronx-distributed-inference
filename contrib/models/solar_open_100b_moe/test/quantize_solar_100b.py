import os
import sys
import gc
import argparse
import torch

os.environ["XLA_HANDLE_SPECIAL_SCALAR"] = "1"
os.environ["UNSAFE_FP8FNCAST"] = "1"

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.application_base import prune_state_dict, save_state_dict_safetensors
from neuronx_distributed.quantization.quantization_config import QuantizationType, QuantizedDtype
from neuronx_distributed.quantization.quantization_utils import (
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
    convert_qint8_to_int8_state_dict,
)
from modeling_solar_open_100b_moe import (
    SolarOpenInferenceConfig,
    NeuronSolarOpenForCausalLM,
    load_solar_open_config,
)


def main():
    parser = argparse.ArgumentParser(description="Solar Open 100B MoE FP8 Quantization")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ubuntu/workspace/model_hf/Solar-Open-100B",
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save quantized checkpoints (auto-generated if not specified)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers to quantize (default: all layers)",
    )
    args = parser.parse_args()

    model_path = args.model_path
    if args.output_path:
        quantized_model_path = args.output_path
    elif args.num_layers:
        quantized_model_path = f"/home/ubuntu/workspace/model_hf/Solar-Open-100B-f8e4m3-{args.num_layers}layer"
    else:
        quantized_model_path = "/home/ubuntu/workspace/model_hf/Solar-Open-100B-f8e4m3"

    neuron_config = MoENeuronConfig(
        quantized=True,
        quantized_checkpoints_path=quantized_model_path,
        quantization_dtype="f8e4m3",
        quantization_type="per_channel_symmetric",
        modules_to_not_convert=["lm_head", "mlp.gate", "o_proj", "shared_experts"],
    )

    config = SolarOpenInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_solar_open_config(model_path),
    )

    if args.num_layers:
        config.num_hidden_layers = args.num_layers

    os.makedirs(quantized_model_path, exist_ok=True)

    num_layers_str = str(args.num_layers) if args.num_layers else "all"
    print(f"Quantizing Solar Open 100B to {quantized_model_path} ...")
    print(f"  layers: {num_layers_str}, dtype: f8e4m3, type: per_channel_symmetric")
    print(f"  modules_to_not_convert: {neuron_config.modules_to_not_convert}")

    print("  Loading HF model ...")
    hf_model = NeuronSolarOpenForCausalLM.load_hf_model(model_path)

    if args.num_layers:
        total = len(hf_model.model.layers)
        print(f"  Truncating model: {total} -> {args.num_layers} layers")
        for i in range(total - 1, args.num_layers - 1, -1):
            del hf_model.model.layers[i]
        gc.collect()

    quantization_type = QuantizationType(config.neuron_config.quantization_type)
    quantized_dtype = QuantizedDtype.get_dtype(config.neuron_config.quantization_dtype)

    print("  Quantizing ...")
    if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
        hf_model_quant = quantize_pytorch_model_per_tensor_symmetric(
            float_model=hf_model, inplace=True, dtype=quantized_dtype,
        )
    elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
        hf_model_quant = quantize_pytorch_model_per_channel_symmetric(
            float_model=hf_model, inplace=True, dtype=quantized_dtype,
            modules_to_not_convert=config.neuron_config.modules_to_not_convert,
        )
    else:
        raise RuntimeError(f"{config.neuron_config.quantization_type} not supported")

    print("  Extracting quantized state dict ...")

    full_sd = hf_model_quant.state_dict()
    model_quant_sd = {}
    for k, v in full_sd.items():
        model_quant_sd[k[len("model."):] if k.startswith("model.") else k] = v
    convert_qint8_to_int8_state_dict(model_quant_sd)
    quantized_state_dict = prune_state_dict(model_quant_sd)

    print(f"  Saving to {quantized_model_path} ...")
    if os.path.isdir(quantized_model_path):
        save_state_dict_safetensors(state_dict=quantized_state_dict, state_dict_dir=quantized_model_path)
    else:
        torch.save(quantized_state_dict, quantized_model_path)

    print("Done.")


if __name__ == "__main__":
    main()
