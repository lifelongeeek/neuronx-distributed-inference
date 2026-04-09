# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import warnings
import math
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.routing import GroupLimitedRouter
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.modules.moe.moe_process_group import (
    init_tensor_expert_parallel_moe_process_groups, get_moe_tp_ep_group, get_moe_ep_group,
)

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.models.layer_boundary_marker import ModuleMarkerEndWrapper, ModuleMarkerStartWrapper

def _rms_norm_cls():
    if cpu_mode():
        return _SimpleRMSNorm
    return CustomRMSNorm

class _SimpleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(self.weight.dtype)

class AdapterSafeRowParallelLinear(RowParallelLinear):
    def forward(self, input_, **kwargs):
        return super().forward(input_)

class NeuronSolarOpenRouter(GroupLimitedRouter):
    def __init__(
        self, num_experts: int, top_k: int, hidden_size: int, n_group: int, topk_group: int,
        norm_topk_prob: bool = True, routed_scaling_factor: float = 1.0,
        sequence_parallel_enabled: bool = False, sequence_dimension: Optional[int] = None,
        dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu"),
        tensor_model_parallel_group=None, jitter_eps: float = 0.0,
    ):
        super().__init__(
            num_experts=num_experts, top_k=top_k, hidden_size=hidden_size, n_group=n_group,
            topk_group=topk_group, sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension, dtype=dtype, device=device,
            tensor_model_parallel_group=tensor_model_parallel_group, jitter_eps=jitter_eps,
        )
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        batch_size, num_experts = router_logits.shape
        
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias.unsqueeze(0)
        
        group_scores = self._calculate_group_scores(scores_for_choice, batch_size)
        group_idx = torch.topk(group_scores, k=self.topk_group)[1]
        group_mask = self._create_group_mask(group_scores, group_idx)
        score_mask = self._expand_group_mask(group_mask, batch_size)
        
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(masked_scores, k=self.top_k)
        
        topk_weights = routing_weights.to(dtype=hidden_states.dtype).gather(1, topk_idx)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        
        full_affinities = torch.zeros_like(router_logits, dtype=hidden_states.dtype)
        full_affinities.scatter_(1, topk_idx, topk_weights)
        topk_idx = topk_idx.detach().to(dtype=torch.long)
        return router_logits, full_affinities, topk_idx

def initialize_solar_open_moe_module(config: "SolarOpenInferenceConfig") -> MoE:
    if config.neuron_config.moe_ep_degree > 1:
        moe_ep_degree = config.neuron_config.moe_ep_degree
        moe_tp_degree = config.neuron_config.moe_tp_degree
        init_tensor_expert_parallel_moe_process_groups(
            moe_tp_degree, moe_ep_degree, moe_tp_degree, moe_ep_degree
        )
        moe_tkg_tp_group = get_moe_tp_ep_group(prefill=False)
        moe_tkg_ep_group = get_moe_ep_group(prefill=False)
        moe_cte_tp_group = get_moe_tp_ep_group(prefill=True)
        moe_cte_ep_group = get_moe_ep_group(prefill=True)
    else:
        moe_tkg_tp_group = parallel_state.get_tensor_model_parallel_group()
        moe_tkg_ep_group = parallel_state.get_expert_model_parallel_group()
        moe_cte_tp_group = parallel_state.get_tensor_model_parallel_group()
        moe_cte_ep_group = parallel_state.get_expert_model_parallel_group()

    router = NeuronSolarOpenRouter(
        num_experts=config.num_local_experts, top_k=config.num_experts_per_tok, hidden_size=config.hidden_size,
        n_group=config.n_group, topk_group=config.topk_group, norm_topk_prob=config.norm_topk_prob,
        routed_scaling_factor=config.routed_scaling_factor, dtype=config.neuron_config.router_config.dtype,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled, sequence_dimension=1,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
    )

    bw_config = config.neuron_config.blockwise_matmul_config
    enable_spmd = bw_config.get("parallelize_token_to_block_mapping", False) if isinstance(bw_config, dict) else getattr(bw_config, "parallelize_token_to_block_mapping", False)

    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts, hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size,
            hidden_size_actual=config.hidden_size, intermediate_size_actual=getattr(config, "original_intermediate_size", config.moe_intermediate_size), 
            is_hidden_dim_shuffled=config.neuron_config.is_hidden_dim_shuffled, is_intermediate_dim_shuffled=config.neuron_config.is_intermediate_dim_shuffled,
            top_k=config.num_experts_per_tok, hidden_act=config.hidden_act, glu_mlp=config.neuron_config.glu_mlp,
            glu_type=config.neuron_config.glu_type, hidden_act_scaling_factor=config.neuron_config.hidden_act_scaling_factor,
            hidden_act_bias=config.neuron_config.hidden_act_bias, use_index_calc_kernel=config.neuron_config.use_index_calc_kernel,
            gate_clamp_upper_limit=config.neuron_config.gate_clamp_upper_limit, gate_clamp_lower_limit=config.neuron_config.gate_clamp_lower_limit,
            up_clamp_upper_limit=config.neuron_config.up_clamp_upper_limit, up_clamp_lower_limit=config.neuron_config.up_clamp_lower_limit,
            normalize_top_k_affinities=False, early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation, enable_spmd_rank=enable_spmd,
        ),
        blockwise_matmul_config=config.neuron_config.blockwise_matmul_config, sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        dtype=config.neuron_config.torch_dtype, is_prefill=config.neuron_config.is_prefill_stage,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(), expert_model_parallel_group=parallel_state.get_expert_model_parallel_group(),
        cte_tensor_model_parallel_group=moe_cte_tp_group, cte_expert_model_parallel_group=moe_cte_ep_group,
        tkg_tensor_model_parallel_group=moe_tkg_tp_group, tkg_expert_model_parallel_group=moe_tkg_ep_group,
    )

    shared_experts = None
    if config.n_shared_experts:
        shared_experts = SharedExperts(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, num_shared_experts=config.n_shared_experts,
            hidden_act=config.hidden_act, dtype=config.neuron_config.torch_dtype, reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            fused_gate_up_projection=config.neuron_config.fused_shared_experts, sequence_parallel_enabled=config.neuron_config.shared_experts_sequence_parallel_enabled,
            transpose_weights=config.neuron_config.transpose_shared_experts_weights,
        )

    moe = MoE(
        router=router, expert_mlps=expert_mlps, shared_experts=shared_experts,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        return_router_logits=config.neuron_config.return_router_logits, sequence_dimension=1,
    )
    moe.eval()
    return moe

class SolarOpenYarnRotaryEmbedding(nn.Module):
    def __init__(self, config: "SolarOpenInferenceConfig"):
        super().__init__()
        
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
            
        self.config = config
        
        if not hasattr(config, "partial_rotary_factor"):
            config.partial_rotary_factor = 1.0
        if not hasattr(config, "head_dim"):
            config.head_dim = config.hidden_size // config.num_attention_heads

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq, attention_scaling = self.rope_init_fn(self.config, device=x.device)

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded * position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class NeuronSolarOpenAttention(NeuronAttentionBase):
    def __init__(self, config: "SolarOpenInferenceConfig"):
        rotary_emb = SolarOpenYarnRotaryEmbedding(config)

        q_layernorm = _rms_norm_cls()(config.head_dim, config.rms_norm_eps) if getattr(config, "use_qk_norm", False) else None
        k_layernorm = _rms_norm_cls()(config.head_dim, config.rms_norm_eps) if getattr(config, "use_qk_norm", False) else None

        super().__init__(
            config=config, hidden_size=config.hidden_size, num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim, rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps, 
            use_qk_norm=False,
            q_layernorm=q_layernorm,
            k_layernorm=k_layernorm,
            qkv_bias=False,
        )

class NeuronSolarOpenDecoderLayer(nn.Module):
    def __init__(self, config: "SolarOpenInferenceConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = NeuronSolarOpenAttention(config=config)
        self.input_layernorm = _rms_norm_cls()(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _rms_norm_cls()(config.hidden_size, config.rms_norm_eps)
        self.mlp = initialize_solar_open_moe_module(config)
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional = None, position_ids: Optional = None,
        past_key_value: Optional = None, padding_mask: Optional = None, **kwargs,
    ) -> Tuple:
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        
        residual = hidden_states
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                normed_hidden_states = hidden_states
            else:
                normed_hidden_states = self.input_layernorm(hidden_states)
        else:
            normed_hidden_states = hidden_states

        attn_out, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=normed_hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_value=past_key_value, 
            **kwargs,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + (mlp_out[0] if isinstance(mlp_out, (tuple, list)) else mlp_out)

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)

class NeuronSolarOpenModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: "SolarOpenInferenceConfig"):
        self.on_device_sampling = (config.neuron_config.on_device_sampling_config is not None)
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: "SolarOpenInferenceConfig"):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.neuron_config.torch_dtype, shard_across_embedding=True)
        self.layers = nn.ModuleList([NeuronSolarOpenDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        
        self.norm = _rms_norm_cls()(config.hidden_size, config.rms_norm_eps)
        
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            sequence_dimension=1,
            tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group() if parallel_state.model_parallel_is_initialized() else None
        )

class NeuronSolarOpenForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronSolarOpenModel
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        return hf_model

    @classmethod
    def get_config_cls(cls):
        return SolarOpenInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: "SolarOpenInferenceConfig") -> dict:
        return convert_solar_open_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
        else:
            optimization_level = "-O1"

        compiler_args = (f"--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer {optimization_level}")
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' --auto-cast=none --internal-enable-dge-levels vector_dynamic_offsets --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args

def load_solar_open_config(model_path: str):
    import json as _json
    from neuronx_distributed_inference.models.config import to_torch_dtype

    def load_config(self: "SolarOpenInferenceConfig"):
        import os as _os
        config_path = _os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config_dict = _json.load(f)

        hf_dtype = config_dict.pop("torch_dtype", config_dict.pop("dtype", None))
        if hf_dtype is not None:
            if (self.neuron_config is not None and not self.neuron_config.overrides_torch_dtype):
                self.neuron_config.torch_dtype = to_torch_dtype(hf_dtype) if isinstance(hf_dtype, str) else hf_dtype

        self.__dict__.update(config_dict)
        if not hasattr(self, "hidden_act"): self.hidden_act = "silu"
        if not hasattr(self, "n_group"): self.n_group = 1
        if not hasattr(self, "topk_group"): self.topk_group = 1
        self._name_or_path = model_path
    return load_config

class SolarOpenInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "output_attentions"): self.output_attentions = False
        if not hasattr(self, "output_hidden_states"): self.output_hidden_states = False
        if not hasattr(self, "is_encoder_decoder"): self.is_encoder_decoder = False
        if not hasattr(self, "hidden_act"): self.hidden_act = "silu"
        if not hasattr(self, "n_group"): self.n_group = 1
        if not hasattr(self, "topk_group"): self.topk_group = 1
        self.num_local_experts = self.n_routed_experts

        if getattr(self.neuron_config.router_config, "dtype", None) is None: self.neuron_config.router_config.dtype = torch.float32
        if getattr(self.neuron_config, "normalize_top_k_affinities", None) is None: self.neuron_config.normalize_top_k_affinities = False
        if getattr(self.neuron_config, "disable_numeric_cc_token", None) is None: self.neuron_config.disable_numeric_cc_token = True
        if getattr(self.neuron_config, "fused_shared_experts", None) is None: self.neuron_config.fused_shared_experts = False
        if getattr(self.neuron_config, "transpose_shared_experts_weights", None) is None: self.neuron_config.transpose_shared_experts_weights = False
        if getattr(self.neuron_config, "shared_experts_sequence_parallel_enabled", None) is None:
            self.neuron_config.shared_experts_sequence_parallel_enabled = getattr(self.neuron_config, "sequence_parallel_enabled", False)
        self.maybe_pad_intermediate()

    def maybe_pad_intermediate(self):
        try: from neuronx_distributed_inference.models.config import SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
        except ImportError: SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP = 256
        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.moe_intermediate_size // moe_tp_degree
        bw_config = self.neuron_config.blockwise_matmul_config
        use_shard = bw_config.get("use_shard_on_intermediate_dynamic_while", False) if isinstance(bw_config, dict) else getattr(bw_config, "use_shard_on_intermediate_dynamic_while", False)
        if use_shard:
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded = (math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP) * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP * moe_tp_degree)
                self.original_intermediate_size = self.moe_intermediate_size
                self.moe_intermediate_size = padded
            else:
                self.original_intermediate_size = self.moe_intermediate_size

    def get_required_attributes(self) -> List[str]:
        return ["head_dim", "hidden_act", "hidden_size", "max_position_embeddings", "moe_intermediate_size", "n_routed_experts", "n_shared_experts", "norm_topk_prob", "num_attention_heads", "num_experts_per_tok", "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_theta", "routed_scaling_factor", "tie_word_embeddings", "vocab_size"]

    @classmethod
    def get_neuron_config_cls(cls): return MoENeuronConfig

def _helper_concat_and_delete_qkv(state_dict: Dict[str, Any], layer_num: int, key_type: str):
    q_key, k_key, v_key = f"layers.{layer_num}.self_attn.q_proj.{key_type}", f"layers.{layer_num}.self_attn.k_proj.{key_type}", f"layers.{layer_num}.self_attn.v_proj.{key_type}"
    if q_key not in state_dict: return
    qkv_key = f"layers.{layer_num}.self_attn.qkv_proj.Wqkv.{key_type}"
    state_dict[qkv_key] = torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0)
    del state_dict[q_key], state_dict[k_key], state_dict[v_key]


def _merge_expert_weights(
    neuron_state_dict: Dict[str, Any],
    layer_idx: int,
    num_experts: int,
    moe_intermediate_size: int,
    is_quantized: bool,
    ep_degree: int = 1,
) -> None:
    l = layer_idx
    gate_proj_0 = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"]
    intermediate_size_e, hidden_size = gate_proj_0.shape
    device, dtype = gate_proj_0.device, gate_proj_0.dtype
    pad_size = max(moe_intermediate_size - intermediate_size_e, 0)

    gate_up_proj = torch.empty(num_experts, hidden_size, 2 * intermediate_size_e, dtype=dtype, device=device)
    down_proj = torch.empty(num_experts, intermediate_size_e, hidden_size, dtype=dtype, device=device)

    has_scales = is_quantized and (f"layers.{l}.mlp.experts.0.gate_proj.scale" in neuron_state_dict)
    if has_scales:
        gate_up_scale = torch.empty(num_experts, 2 * intermediate_size_e, dtype=torch.float32, device=device)
        down_scale = torch.empty(num_experts, hidden_size, dtype=torch.float32, device=device)

    for e in range(num_experts):
        gate_w = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.gate_proj.weight").T.detach().clone()
        up_w = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.up_proj.weight").T.detach().clone()
        down_w = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.down_proj.weight").T.detach().clone()

        gate_up_slice = torch.narrow(gate_up_proj, 0, e, 1)
        torch.narrow(gate_up_slice, 2, 0, intermediate_size_e).copy_(gate_w)
        torch.narrow(gate_up_slice, 2, intermediate_size_e, intermediate_size_e).copy_(up_w)
        torch.narrow(down_proj, 0, e, 1).copy_(down_w)

        if has_scales:
            gate_s = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.gate_proj.scale").detach().clone()
            up_s = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.up_proj.scale").detach().clone()
            down_s = neuron_state_dict.pop(f"layers.{l}.mlp.experts.{e}.down_proj.scale").detach().clone()
            gate_up_scale[e] = torch.cat([gate_s.flatten(), up_s.flatten()])
            down_scale[e] = down_s.flatten()

    if pad_size > 0:
        gate_up_proj = torch.nn.functional.pad(
            gate_up_proj.reshape(num_experts, hidden_size, 2, intermediate_size_e), (0, pad_size)
        ).reshape(num_experts, hidden_size, -1)
        down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))
        if has_scales:
            gate_up_scale = torch.nn.functional.pad(
                gate_up_scale.reshape(num_experts, 2, intermediate_size_e), (0, pad_size), value=1.0
            ).reshape(num_experts, -1)
    else:
        gate_up_proj = gate_up_proj.reshape(num_experts, hidden_size, -1)

    neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
    neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

    if has_scales:
        experts_per_ep = num_experts // ep_degree
        fp8_dtype = gate_up_proj.dtype

        for name, weight, scale_2d in [
            ("gate_up_proj", f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight", gate_up_scale),
            ("down_proj", f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight", down_scale),
        ]:
            w = neuron_state_dict[weight]  # [E, *, dim]  fp8
            # EP max scale: [ep, dim]
            ep_max_scale = scale_2d.reshape(ep_degree, experts_per_ep, -1).max(dim=1).values

            for ep_idx in range(ep_degree):
                ep_start = ep_idx * experts_per_ep
                ep_end = ep_start + experts_per_ep
                max_s = ep_max_scale[ep_idx]  # [dim]

                for e in range(ep_start, ep_end):
                    expert_s = scale_2d[e]  # [dim]
                    # dequant → requant with max_scale
                    float_w = w[e].to(torch.float32) * expert_s.unsqueeze(0)
                    rescaled = float_w / max_s.unsqueeze(0)
                    w[e] = rescaled.clamp(-240, 240).to(fp8_dtype)

            neuron_state_dict[weight] = w
            neuron_state_dict[weight.replace(".weight", ".scale")] = ep_max_scale.unsqueeze(1)


def convert_solar_open_hf_to_neuron_state_dict(neuron_state_dict: Dict[str, Any], config: "SolarOpenInferenceConfig") -> Dict[str, Any]:
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    is_quantized = getattr(config.neuron_config, "quantized", False)
    _per_expert_format = f"layers.0.mlp.experts.0.gate_proj.weight" in neuron_state_dict
    neuron_state_dict["rank_util.rank"] = torch.arange(0, config.neuron_config.tp_degree, dtype=torch.int32)
    num_moe_experts = config.n_routed_experts

    for l in range(config.num_hidden_layers):
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(0, config.neuron_config.tp_degree, dtype=torch.int32)

        for suffix in ["weight", "scale"]:
            o_key = f"layers.{l}.self_attn.o_proj.{suffix}"
            if o_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.self_attn.o_proj.o_proj.{suffix}"] = neuron_state_dict.pop(o_key)

        if not config.neuron_config.fused_qkv:
            for suffix in ["weight", "scale"]:
                for proj in ["q_proj", "k_proj", "v_proj"]:
                    proj_key = f"layers.{l}.self_attn.{proj}.{suffix}"
                    if proj_key in neuron_state_dict:
                        neuron_state_dict[f"layers.{l}.self_attn.qkv_proj.{proj}.{suffix}"] = neuron_state_dict.pop(proj_key)

        if getattr(config, "use_qk_norm", False):
            q_norm_key = f"layers.{l}.self_attn.q_norm.weight"
            k_norm_key = f"layers.{l}.self_attn.k_norm.weight"
            if q_norm_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = neuron_state_dict.pop(q_norm_key).detach().clone()
            if k_norm_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = neuron_state_dict.pop(k_norm_key).detach().clone()

        gate_weight_key = f"layers.{l}.mlp.gate.weight"
        if gate_weight_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = neuron_state_dict.pop(gate_weight_key).detach().clone()

        bias_key = f"layers.{l}.mlp.gate.e_score_correction_bias"
        if bias_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.router.e_score_correction_bias"] = neuron_state_dict.pop(bias_key).detach().clone().to(torch.float32)

        if _per_expert_format:
            _merge_expert_weights(
                neuron_state_dict, l, num_moe_experts,
                config.moe_intermediate_size, is_quantized,
                ep_degree=config.neuron_config.moe_ep_degree,
            )
        else:
            gate_up_key = f"layers.{l}.mlp.experts.gate_up_proj"
            if gate_up_key in neuron_state_dict:
                gate_up_neuron = neuron_state_dict[gate_up_key].permute(0, 2, 1).detach().clone()
                E, H, two_I = gate_up_neuron.shape
                I, pad_size = two_I // 2, max(config.moe_intermediate_size - (two_I // 2), 0)

                gate_up_neuron = gate_up_neuron.reshape(E, H, 2, I)
                if pad_size > 0: gate_up_neuron = torch.nn.functional.pad(gate_up_neuron, (0, pad_size))
                neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_neuron.reshape(E, H, -1)
                del neuron_state_dict[gate_up_key]

            down_key = f"layers.{l}.mlp.experts.down_proj"
            if down_key in neuron_state_dict:
                down_neuron = neuron_state_dict[down_key].permute(0, 2, 1).detach().clone()
                E, I_dim, H = down_neuron.shape
                pad_size = max(config.moe_intermediate_size - I_dim, 0)
                if pad_size > 0: down_neuron = torch.nn.functional.pad(down_neuron, (0, 0, 0, pad_size))
                neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_neuron
                del neuron_state_dict[down_key]

        for proj in ["gate_proj", "up_proj"]:
            proj_key = f"layers.{l}.mlp.shared_experts.{proj}.weight"
            scale_key = f"layers.{l}.mlp.shared_experts.{proj}.scale"
            if proj_key in neuron_state_dict:
                w = neuron_state_dict[proj_key]
                if len(w.shape) > 1:
                    shared_pad_size = max(config.intermediate_size - w.shape[0], 0)
                    if shared_pad_size > 0:
                        neuron_state_dict[proj_key] = torch.nn.functional.pad(w, (0, 0, 0, shared_pad_size))
                        if scale_key in neuron_state_dict:
                            s = neuron_state_dict[scale_key]
                            neuron_state_dict[scale_key] = torch.nn.functional.pad(s, (0, 0, 0, shared_pad_size), value=1.0)

        down_key = f"layers.{l}.mlp.shared_experts.down_proj.weight"
        if down_key in neuron_state_dict:
            w = neuron_state_dict[down_key]
            if len(w.shape) > 1:
                shared_pad_size = max(config.intermediate_size - w.shape[1], 0)
                if shared_pad_size > 0:
                    w = torch.nn.functional.pad(w, (0, shared_pad_size))
            neuron_state_dict[down_key] = w

    keys_to_delete = []
    for key in list(neuron_state_dict.keys()):
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) >= config.num_hidden_layers:
                keys_to_delete.append(key)

    for key in keys_to_delete:
        if key in neuron_state_dict: del neuron_state_dict[key]
    gc.collect()

    if config.neuron_config.fused_qkv:
        for l in range(config.num_hidden_layers):
            _helper_concat_and_delete_qkv(neuron_state_dict, l, "weight")
            if is_quantized:
                _helper_concat_and_delete_qkv(neuron_state_dict, l, "scale")

    return neuron_state_dict