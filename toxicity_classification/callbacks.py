from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class ValidationLossEarlyStopping(EarlyStopping):
    def __init__(self, experiment_params: "dict"):
        super().__init__(monitor="validation_loss",
                         min_delta=experiment_params['early_stopping']["validation_loss"]["min_delta"],
                         patience=experiment_params['early_stopping']["validation_loss"]["patience"], mode="min")
        self.experiment_params = experiment_params

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.experiment_params['early_stopping']["validation_loss"]["to_check"] is True:
            self._run_early_stopping_check(trainer)


class TrainingLossEarlyStopping(EarlyStopping):
    def __init__(self, experiment_params: "dict"):
        super().__init__(monitor="training_loss",
                         min_delta=experiment_params['early_stopping']["training_loss"]["min_delta"],
                         patience=experiment_params['early_stopping']["training_loss"]["patience"], mode="min")
        self.experiment_params = experiment_params

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.experiment_params['early_stopping']["training_loss"]["to_check"] is True:
            self._run_early_stopping_check(trainer)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass