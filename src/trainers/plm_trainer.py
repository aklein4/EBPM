import torch
import torch.nn as nn
import torch.nn.functional as F

from models.plm.modelling_plm import ProteinLM
from trainers.base_trainer import BaseTrainer


class PLMTrainer(BaseTrainer):
    
    model: ProteinLM


    def train_forward(
        self,
        step,
        input_ids,
    ):
        
        logits = self.model.forward(
            input_ids=input_ids
        ).logits

        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )

        return lm_loss, {
            "lm_loss": lm_loss,
        }
    