import torch
import pytorch_lightning as pl

from models import ToxicCommentTaggerBERT


class ToxicityClassificationTrainer(pl.LightningModule):
    def __init__(self, experiment_params: dict):
        super().__init__()
        self.experiment_params = experiment_params
        self.toxicity_model = ToxicCommentTaggerBERT(experiment_params)
        self.lowest_valid_loss = float("inf")

    def common_step(self, batch):
        return batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], batch["labels"]

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        logits = self.toxicity_model(input_ids, attention_mask, token_type_ids)
        probabilities = torch.sigmoid(logits)
        loss = 0
        if labels is not None:
            loss = self.loss_fn(probabilities, labels)
        return loss, probabilities

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

    def configure_optimizers(self):
        optimizer = self.experiment_params["optimizer"]["class"](self.parameters(),
                                                                 self.experiment_params["optimizer"]["kwargs"])

        if "scheduler" in self.experiment_params.keys():
            scheduler = self.experiment_params["scheduler"]["class"](self.experiment_params["scheduler"]["kwargs"])
            return dict(optimizer=optimizer, lr_scheduler=scheduler)

        return dict(optimizer=optimizer)

    def training_epoch_end(self, outputs):
        """
        This function is called at the end of each training epoch
        :param outputs:
        :return:
        """
        loss = sum(output['loss'] for output in outputs)/len(outputs)
        self.log("Training loss", loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        """
        This function is called at the end of
        :param outputs:
        """
        validation_loss = sum(output['loss'] for output in outputs)/len(outputs)
        self.log("Validation loss", validation_loss, prog_bar=True, logger=True)

        if validation_loss < self.lowest_valid_loss:
            self.lowest_valid_loss = validation_loss
            torch.save(self.model.state_dict())