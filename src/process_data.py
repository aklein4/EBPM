
import datasets

import numpy as np


IN_URL = "nvidia/esm2_uniref_pretraining_data"
OUT_URL = "aklein4/esm2_uniref_pretraining_data-512-tokenized"

LENGTH = 512


def process_split(split):

    data = datasets.load_dataset(IN_URL, split=split, streaming=False)

    def map_fn(batch):
        batch["sequence"] = [s[:LENGTH] for s in batch["sequence"]]
        return batch
    data = data.map(map_fn, batched=True, batch_size=1000)

    data = data.filter(lambda x: len(x["sequence"]) == LENGTH)

    def tokenize_fn(batch):

        input_ids = []
        for raw in batch["sequence"]:
            s = raw.upper()

            ids = []
            for char in s:

                id_ = 0
                if char.isalpha():
                    id_ = ord(char.upper()) - ord('A') + 1
                
                ids.append(id_)
            input_ids.append(np.array(ids, dtype="uint8"))

        return {"input_ids": input_ids}
        
    data = data.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
    )

    data.push_to_hub(OUT_URL, split=split)


def main():
    
    for split in datasets.get_dataset_split_names(IN_URL):
        process_split(split)


if __name__ == "__main__":
    main()