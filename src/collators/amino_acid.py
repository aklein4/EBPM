import torch

import utils.constants as constants


class AminoAcidCollator:

    def __init__(
        self,
    ):
        return
    

    def __call__(
        self,
        raw
    ):
        
        input_ids = []
        for example in raw:
            s = example["sequence"].upper()

            ids = []
            for char in s:

                id_ = 0
                if char.isalpha():
                    id_ = ord(char.upper()) - ord('A') + 1
                
                ids.append(id_)
            input_ids.append(ids)

        input_ids = torch.tensor(input_ids).long().to(constants.DEVICE)

        return {
            "input_ids": input_ids
        }
    