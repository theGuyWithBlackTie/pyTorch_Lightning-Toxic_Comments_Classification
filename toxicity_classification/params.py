import torch.nn as nn
import torch.optim
import transformers

experiment_params = {
    "tokenizer_name": "google-bert/bert-base-uncased",
    "transformer_name": "google-bert/bert-base-uncased",
    "ffn_params": {
        "layers": [768, 384, 7],
        "activation":[nn.ReLU(), nn.ReLU()],
        "dropout": [0.5]
    },
    "optimizer":{
    "adam": {
        "class": torch.optim.Adam,
        "kwargs": {
            "weight_decay": 5e-4,
            "lr": 1e-5,
            "betas": (0.9, 0.999)
            }
        },
    "adamw": {
        "class": torch.optim.AdamW,
        "kwargs": {
            "lr": 1e-6,
            "betas": (0.9, 0.999),
            "weight_decay": 1e-5
            }
        }
    },
    "scheduler": {
        "get-linear-schedule-with-warmup": {
            "class": transformers.get_linear_schedule_with_warmup,
            "compute": ["num_warmup_steps", "num_training_steps"],
            "kwargs": {}
            },

        "cosine-annealing": {
            "class": transformers.get_cosine_with_hard_restarts_schedule_with_warmup,
            "compute": ["num_warmup_steps", "num_training_steps"],
            "kwargs":{
                "num_cycles": 1
                }
            }
    },
    "dataset_params": {
        "train_data_path": "../data/train_dataset.pkl",
        "val_data_path": "../data/validation_dataset.pkl",
        "test_data_path": "../data/test_dataset.pkl",
        "labels": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "no_toxicity"]
    },
    "epochs": 30,
    "warmup_proportion": 0.1,
    "early_stopping": {
        "validation_loss": {
            "to_check": True,
            "patience": 20,
            "min_delta": 0.001
        },
        "training_loss": {
            "to_check": False,
            "patience": 20,
            "min_delta": 0.001
        }
    }
}