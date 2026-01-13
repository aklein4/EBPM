import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

from models.ebpm.modelling_ebpm import EBPMModel
from utils.torch_utils import shift


DEVICE = "cpu"

SEQUENCE = "MTTISLRAALLGASVALLAPIGLAVAADGSGYRIEAVTLSAGGLAEIRRGVQVDGASDLGFDVPLDQVSDILKSLLVYDAAGGVASIRLDGPSPVEETFRGLPFTPEDMNGLPSLLKTLQGTSVRVTSGGRTVEGMVMGVAEDSQPAKDDSERGQLLSVMTAEGQIAVLRLRSDTQLDILDVAMRDKLRAAATVSGKSRVEDMRTINVGLEGTGERDVFLDYVVPAPIWKTAYRLMLDADGKARLQAWAVIENATGEDWSNVAITLSSGAPVTLSQRLYERYWHERQDVPVLAQSVMAPPPDLYKGGAVDERSRLANQDMAYEMMAVPAQAVYAPAPISMSPSAPVAQATASDGQTAAIYRLPMPVDLGAGQTLSVPYIDTTLDAERIAIFYPDRGDTHPISALKLENATGTSLPPGIVTVYAPQEEGYAGDAQLMGVPSAESRILSFAADRKVEVTTERGNEQTSYRATIANGVLRIISTTRADTTYTIKGAPDASRTVIIEHPRLDGWTF"

CHECKPOINT = "/home/ubuntu/EBPM/src/local_data/EBPM_alpha/step_000000002000"


def tokenize_sequence(seq):

    tokens = []
    for char in seq.upper():

        if char.isalpha():
            tokens.append(ord(char) - ord("A") + 1)
        else:
            tokens.append(0)
    
    return torch.tensor(tokens, dtype=torch.long).to(DEVICE)
        

@torch.no_grad()
def main():
    
    model = EBPMModel.from_pretrained(CHECKPOINT).to(DEVICE)
    model.eval()
    print("Model loaded.")

    input_ids = tokenize_sequence(SEQUENCE).unsqueeze(0)
    print("Input tokenized.")

    logits, mem, kv_states = model.base_forward(
        input_ids
    )
    print("Base forward done.")

    dist = torch.distributions.Categorical(logits=logits)

    energies = []
    for i in tqdm(range(10)):

        negatives = dist.sample()
        negatives = shift(negatives, n=1, dim=1, direction="left", narrow=True)

        energy = model.ebm_forward(
            negatives, mem, kv_states
        )[:, :-1]
        energies.append(energy.squeeze(0))

    energies = torch.stack(energies, dim=0)
    z = energies.exp().mean(dim=0)

    plt.hist(z.cpu().numpy(), bins=20)
    plt.savefig("z_hist.png")

    print("Average partition function z:", z.mean().item())


if __name__ == "__main__":
    main()