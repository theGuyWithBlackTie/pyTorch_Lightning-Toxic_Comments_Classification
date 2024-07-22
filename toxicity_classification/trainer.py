import torch
import pytorch_lightning as pl
from torchmetrics.functional.classification import multilabel_f1_score

from models import ToxicCommentTaggerBERT


class ToxicityClassificationTrainer(pl.LightningModule):
    def __init__(self, experiment_params: dict):
        super().__init__()
        self.experiment_params = experiment_params
        self.toxicity_model = ToxicCommentTaggerBERT(experiment_params)
        self.lowest_valid_loss = float("inf")

        self.training_step_loss_outputs = []
        self.validation_step_loss_outputs = []
        self.test_step_loss_outputs = []

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def common_step(self, batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "labels": batch["labels"]
        }

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        logits = self.toxicity_model(input_ids, attention_mask, token_type_ids)
        probabilities = torch.sigmoid(logits)
        loss = 0
        if labels is not None:
            loss = self.loss_fn(probabilities, labels)
        else:
            labels = None
        return loss, probabilities, labels

    def training_step(self, batch, batch_idx):
        loss, probabilities, labels = self(**self.common_step(batch))
        self.log("training_loss", loss, prog_bar=True, logger=False)
        self.training_step_loss_outputs.append(loss)
        self.training_F1_score = multilabel_f1_score(probabilities, labels,
                                                     num_labels=len(self.experiment_params["dataset_params"]["labels"]))
        self.log("training_F1_score", self.training_F1_score, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probabilities, labels = self(**self.common_step(batch))
        self.log("validation_loss", loss, prog_bar=True, logger=False)
        self.validation_step_loss_outputs.append(loss)
        self.validation_F1_score = multilabel_f1_score(probabilities, labels,
                                                       num_labels=len(self.experiment_params["dataset_params"]["labels"]))
        self.log("validation_F1_score", self.validation_F1_score, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, probabilities, labels = self(**self.common_step(batch))
        self.log("test_loss", loss, prog_bar=True, logger=False)
        self.test_step_loss_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.experiment_params["optimizer"]["class"](self.parameters(),
                                                                 **self.experiment_params["optimizer"]["kwargs"])

        if "scheduler" in self.experiment_params.keys():
            num_warmup_steps = int(self.experiment_params["dataset_params"]["train_dataloader_length"]*
                                self.experiment_params["epochs"]*self.experiment_params["warmup_proportion"])
            total_training_steps = int(self.experiment_params["dataset_params"]["train_dataloader_length"] *
                                       self.experiment_params["epochs"])

            if "num_warmup_steps" in self.experiment_params["scheduler"]["compute"]:
                self.experiment_params["scheduler"]["kwargs"]["num_warmup_steps"] = num_warmup_steps
            if "num_training_steps" in self.experiment_params["scheduler"]["compute"]:
                self.experiment_params["scheduler"]["kwargs"]["num_training_steps"] = total_training_steps

            scheduler = self.experiment_params["scheduler"]["class"](optimizer,
                                                                       **self.experiment_params["scheduler"]["kwargs"])
            return dict(optimizer=optimizer, lr_scheduler=scheduler)

        return dict(optimizer=optimizer)

    def on_train_epoch_end(self):
        """
        This function is called at the end of each training epoch
        :param outputs:
        :return:
        """
        training_loss = torch.stack(self.training_step_loss_outputs).mean()
        self.log("training_loss", training_loss, prog_bar=True, logger=True)
        self.training_step_loss_outputs.clear()

    def on_validation_epoch_end(self):
        """
        This function is called at the end of validation epoch
        :param outputs:
        """
        validation_loss = torch.stack(self.validation_step_loss_outputs).mean()
        self.log("validation_loss", validation_loss, prog_bar=True, logger=True)
        self.validation_step_loss_outputs.clear()

        if validation_loss < self.lowest_valid_loss:
            self.lowest_valid_loss = validation_loss
            torch.save(self.toxicity_model.state_dict(), "experiment_outputs/toxicity_model.pth")
