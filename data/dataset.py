from datasets import load_dataset
from transformers import AutoTokenizer
import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, max_len=64):
        self.dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train[:1%]"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return tokens.input_ids.squeeze(0)