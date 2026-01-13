
from models.reference_llama.configuration_llama import LlamaConfig


class EBPMConfig(LlamaConfig):

    model_type = "ebpm"

    def __init__(
        self,
        first_ebm_layer=0,
        lora_rank=4,
        lora_init_scale=0.1,
        ebm_head_init_scale=0.1,
        **kwargs,
    ):
        """
        Configuration class for EBPM model.
        
        Inherits from LlamaConfig.
        """

        self.first_ebm_layer = first_ebm_layer

        self.lora_rank = lora_rank
        self.lora_init_scale = lora_init_scale

        self.ebm_head_init_scale = ebm_head_init_scale

        super().__init__(**kwargs)
        