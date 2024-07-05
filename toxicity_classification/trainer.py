import torch
from torch import nn

import pytorch_lightning as pl

from transformers import AutoModel


class ToxicCommentTaggerBERT(nn.Module):
    def __init__(self, experiment_params: dict):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(experiment_params["transformer_name"])
        self.classification_layers = nn.ModuleList()
        for index in range(0, len(experiment_params["ffn_params"]["layers"]) - 2):
            self.classification_layers.append(nn.Linear(experiment_params["ffn_params"]["layers"][index],
                                                        experiment_params["ffn_params"]["layers"][index + 1]))
            self.classification_layers.append(experiment_params["ffn_params"]["activation"][index])

            if "dropout" in experiment_params["ffn_params"].keys():
                self.classification_layers.append(nn.Dropout(experiment_params["ffn_params"]["dropout"][index]))

        self.classification_layers.append(experiment_params["ffn_params"]["activation"][-1])
        self.classification_layers.append(
            nn.Linear(experiment_params["ffn_params"]["layers"][-2], experiment_params["ffn_params"]["layers"][-1]))

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output_layer = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        logits = self.classification_layers(output_layer)
        return logits


class ToxicityClassificationTrainer(pl.LightningModule):
    def __init__(self, experiment_params: dict):
        super().__init__()
        self.experiment_params = experiment_params
        self.toxicity_model = ToxicCommentTaggerBERT(experiment_params)
        self.loss_fn = nn.BCELoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        logits = self.toxicity_model(input_ids, attention_mask, token_type_ids)
        probabilities = torch.sigmoid(logits)
        loss = 0
        if labels is not None:
            loss = self.loss_fn(probabilities, labels)
        return loss, probabilities

    def common_step(self, batch):
        return batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], batch["labels"]

    def training_step(self, batch, batch_idx):
        loss, probabilities = self(self.common_step(batch))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probabilities = self(self.common_step(batch))
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, probabilities = self(self.common_step(batch))
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss