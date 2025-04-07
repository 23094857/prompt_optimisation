from typing import Any
from datasets import load_from_disk
import random

class DataLoader():

    _dataset_full = None

    def __init__(self, dataset_name, seed=None, mode="train", split=0.8) -> None:  
        self.dataset_name = dataset_name
        self.dataset_mode = mode

        if dataset_name == "math" and (DataLoader._dataset_full is None or DataLoader._dataset_full["name"] != dataset_name):
            dataset = load_from_disk("./data_sets/math_dataset_algLin")

            if seed is None:
                seed = random.getrandbits(32)

            dataset = dataset.shuffle(seed=seed)

            DataLoader._dataset_full = {"name": dataset_name, "dataset": dataset}

        else:
            dataset = DataLoader._dataset_full["dataset"]

        idx_split = int(split *  len(dataset))

        if mode == "train":
            self.dataset = dataset.select(range(idx_split))
        elif mode == "test":
            self.dataset = dataset.select(range(idx_split, len(dataset)))
        else:
            raise ValueError(f"Invalid mode: {mode}. Should be 'train' or 'test'")
        
    
    def __getitem__(self, __name: int) -> Any:
        if self.dataset_name == "math":
            return {'question':self.dataset[__name]['question'].split("b'")[1].split(r"\n")[0],
                    'answer':self.dataset[__name]['answer'].split("b'")[1].split(r"\n")[0]}
        
    def __len__(self) -> int:
        return len(self.dataset)