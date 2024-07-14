import argparse
from pytorch_lightning import Trainer

from params import experiment_params
from trainer import ToxicityClassificationTrainer
from dataset import ToxicCommentsDataModule

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


    trainer = Trainer()
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
