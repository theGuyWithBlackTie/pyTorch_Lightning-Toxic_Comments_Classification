import argparse
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from params import experiment_params
from trainer import ToxicityClassificationTrainer
from dataset import ToxicCommentsDataModule
from callbacks import ValidationLossEarlyStopping, TrainingLossEarlyStopping


def return_early_stopping_callback(patience, min_delta):
    return EarlyStopping(monitor="validation_loss", patience=patience, min_delta=min_delta, verbose=True, mode="min")

def get_logger():
    return MLFlowLogger(experiment_name="Toxicity Classification",
                        run_name=f"toxicity_classification_{str(datetime.now())}",
                        tracking_uri="http://localhost:5000", artifact_location="./artifacts")

if __name__ == "__main__":
    # -----------------------------------
    # Reading the command line arguments
    # -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", help="Optimizer to use", default="adam")
    parser.add_argument("--scheduler", help="Scheduler to use", default="cosine-annealing")
    args = parser.parse_args()
    experiment_params["optimizer"] = experiment_params["optimizer"][args.optimizer]
    experiment_params["scheduler"] = experiment_params["scheduler"][args.scheduler]

    training_loss_earlystopping = TrainingLossEarlyStopping(experiment_params)
    validation_loss_earlystopping = ValidationLossEarlyStopping(experiment_params)
    trainer = Trainer(callbacks=[training_loss_earlystopping, validation_loss_earlystopping],
                      max_epochs=experiment_params["epochs"], logger=get_logger())

    data_module = ToxicCommentsDataModule(
        train_data_path=experiment_params["dataset_params"]["train_data_path"],
        val_data_path=experiment_params["dataset_params"]["val_data_path"],
        test_data_path=experiment_params["dataset_params"]["test_data_path"],
        labels=experiment_params["dataset_params"]["labels"],
        tokenizer_name=experiment_params["tokenizer_name"]
    )
    data_module.setup()

    experiment_params["dataset_params"]["train_dataloader_length"] = len(data_module.train_dataloader())
    model = ToxicityClassificationTrainer(experiment_params)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
