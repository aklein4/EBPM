
import datasets


IN_URL = "nvidia/esm2_uniref_pretraining_data"
OUT_URL = "aklein4/esm2_uniref_pretraining_data-512"

LENGTH = 512


def process_split(split):

    data = datasets.load_dataset(IN_URL, split=split, streaming=False)

    def map_fn(example):
        example["sequence"] = [s[:LENGTH] for s in example["sequence"]]
        return example
    data = data.map(map_fn, batched=True, batch_size=1000)

    data = data.filter(lambda x: len(x["sequence"]) == LENGTH)

    data.push_to_hub(OUT_URL, split=split)


def main():
    
    for split in datasets.get_dataset_split_names(IN_URL):
        process_split(split)


if __name__ == "__main__":
    main()