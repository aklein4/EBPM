import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from models.custom_llama.modelling_llama import (
    LlamaRMSNorm,
    CustomLlamaDecoderLayer,
    CustomLlamaPreTrainedModel,
    CustomLlamaModel
)
from models.ebpm.configuration_ebpm import EBPMConfig


class LoRALayer(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
        init_scale=1.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features)
            )
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(
            math.sqrt(init_scale) * torch.randn(rank, in_features) / math.sqrt(in_features)
        )
        self.lora_B = nn.Parameter(
            math.sqrt(init_scale) * torch.randn(out_features, rank) / math.sqrt(rank)
        )
        self._lora_enabled = False


    def enable_lora(self):
        self._lora_enabled = True
    
    def disable_lora(self):
        self._lora_enabled = False
    
    def is_lora_enabled(self):
        return self._lora_enabled

    def set_lora_enabled(self, enabled: bool):
        self._lora_enabled = enabled


    def forward(self, x):

        w = self.weight
        if self._lora_enabled:
            w = w + self.lora_B @ self.lora_A
        
        return F.linear(x, w, self.bias)


class EBPMDecoderLayer(CustomLlamaDecoderLayer):

    pre_forward_kwargs = ["ebm_mode"]
    post_forward_kwargs = ["ebm_mode", "kv_states"]


    def post_init(self, config: EBPMConfig, layer_idx: int):

        self.layer_idx = layer_idx
        self.is_ebm_layer = (layer_idx >= config.first_ebm_layer)

        if self.is_ebm_layer:
            
            for module in list(self.modules()):
                for name, child in module.named_children():

                    if (
                        isinstance(child, nn.Linear) and
                        "k_proj" not in name and
                        "v_proj" not in name
                    ):

                        lora_layer = LoRALayer(
                            in_features=child.in_features,
                            out_features=child.out_features,
                            rank=config.lora_rank,
                            bias=(child.bias is not None),
                            init_scale=config.lora_init_scale,
                        )

                        setattr(module, name, lora_layer)


    def set_lora_enabled(self, enabled: bool):
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.set_lora_enabled(enabled)


    def pre_forward(self, hidden_states, ebm_mode):
        self.set_lora_enabled(ebm_mode)

        if ebm_mode:
            return hidden_states, {}
    
        return hidden_states, {"incoming_residual": hidden_states.clone()}
    

    def post_forward(self, hidden_states, ebm_mode, kv_states):
        self.set_lora_enabled(False)

        if ebm_mode:
            return hidden_states, {}
        
        return hidden_states, {"kv_states": kv_states}


class EBPMTransformer(CustomLlamaModel):

    layer_type = EBPMDecoderLayer

    skip_norm = True

    gaussian_initialization = True


class EBPMModel(CustomLlamaPreTrainedModel):

    gaussian_initialization = True

    transformer_type = EBPMTransformer


    def __init__(self, config: EBPMConfig):
        super().__init__(config)
        
        self.model = self.transformer_type(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.first_ebm_layer = config.first_ebm_layer
        self.ebm_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.ebm_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ebm_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.ebm_head.weight.data.mul_(config.ebm_head_init_scale)

    
    def base_forward(
        self,
        input_ids,
    ):
        
        outputs = self.model(
            input_ids=input_ids,
            ebm_mode=False,
        )

        hidden_states = self.model.norm(outputs.last_hidden_state)
        logits = self.lm_head(hidden_states)

        mem = outputs.results["incoming_residual"][self.first_ebm_layer]
        kv_states = outputs.results["kv_states"]

        return logits, mem, kv_states
    

    def ebm_forward(
        self,
        sampled_ids,
        mem,
        kv_states,
    ):
        
        inputs_embeds = (
            mem +
            self.ebm_proj(self.model.embed_tokens(sampled_ids))
        )
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            ebm_mode=True,
            layer_slice=slice(self.first_ebm_layer, None),
            kv_states=kv_states,
        )

        hidden_states = self.ebm_norm(outputs.last_hidden_state)
        energy = self.ebm_head(self.ebm_proj(hidden_states)).squeeze(-1)

        return energy
    