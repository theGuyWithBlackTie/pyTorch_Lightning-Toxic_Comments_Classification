import argparse
from pytorch_lightning import Trainer

from params import experiment_params
from trainer import ToxicityClassificationTrainer

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
    model = ToxicityClassificationTrainer(experiment_params)
    trainer.fit(model)
