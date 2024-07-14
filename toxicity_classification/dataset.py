import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

import pickle

class ToxicCommentsDataset(Dataset):
    def __init__(self, data_path: str, labels: list, tokenizer_name: str = 'bert-base-uncased', max_length: int = 256):
        super(ToxicCommentsDataset, self).__init__()

        with open(data_path, 'rb') as fObj:
            self.dataset = pickle.load(fObj)

        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        inputs = self.tokenizer.encode_plus(self.dataset[index]['comment'],
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True)

        labels_tensor = []
        for each_label in self.labels:
            labels_tensor.append(self.dataset[index][each_label])

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(labels_tensor, dtype=torch.float)
        }


class ToxicCommentsDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path: str, val_data_path: str, test_data_path: str, labels: list,
                 tokenizer_name: str = 'bert-base-uncased', max_length: int = 256, batch_size: int = 16):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.labels = labels
        self.tokenizer = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size

    def setup(self, stage: str = None) -> None:
        self.train_dataset = ToxicCommentsDataset(self.train_data_path, self.labels, self.tokenizer, self.max_length)
        self.val_dataset = ToxicCommentsDataset(self.val_data_path, self.labels, self.tokenizer, self.max_length)
        self.test_dataset = ToxicCommentsDataset(self.test_data_path, self.labels, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)