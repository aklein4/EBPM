import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ebpm.modelling_ebpm import EBPMModel
from trainers.base_trainer import BaseTrainer
from utils.torch_utils import shift


class EBPMTrainer(BaseTrainer):
    
    model: EBPMModel


    def train_forward(
        self,
        step,
        input_ids,
    ):
        
        logits, mem, kv_states = self.model.base_forward(
            input_ids
        )

        dist = torch.distributions.Categorical(logits=logits)
        negatives = dist.sample()

        positive_energy = self.model.ebm_forward(
            shift(input_ids, n=1, dim=1, direction="left", narrow=True),
            mem, kv_states
        )[:, :-1]
        negative_energy = self.model.ebm_forward(
            shift(negatives, n=1, dim=1, direction="left", narrow=True),
            mem, kv_states
        )[:, :-1]

        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        energy_loss = (
            -positive_energy.mean() +
            negative_energy.exp().mean()
        )

        loss = lm_loss + energy_loss

        nlogp = lm_loss - positive_energy.mean()
        energy_acc = (
            (positive_energy > 0).float().mean() +
            (negative_energy < 0).float().mean()
        ) / 2

        return loss, {
            "lm_loss": lm_loss,
            "energy_loss": energy_loss,
            "nlogp": nlogp,
            "energy_acc": energy_acc,
            "positive_energy": positive_energy.mean(),
            "negative_energy": negative_energy.mean(),
        }
    