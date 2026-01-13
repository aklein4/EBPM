
from models.plm.configuration_plm import ProteinLMConfig
from models.custom_llama.modelling_llama import CustomLlamaForCausalLM


class ProteinLM(CustomLlamaForCausalLM):
    
    config: ProteinLMConfig

    gaussian_initialization = True
