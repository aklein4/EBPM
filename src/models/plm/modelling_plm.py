
from models.plm.configuration_plm import ProteinLMConfig
from models.custom_llama.modelling_llama import CustomLlamaForCausalLM, CustomLlamaModel


class ProteinTransformer(CustomLlamaModel):

    gaussian_initialization = True


class ProteinLM(CustomLlamaForCausalLM):
    
    config: ProteinLMConfig

    transformer_type = ProteinTransformer

    gaussian_initialization = True
