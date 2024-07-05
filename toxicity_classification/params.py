import torch.nn as nn
import torch.optim

experiment_params = {
    "tokenizer_type": "google-bert/bert-base-uncased",
    "transformer_name": "google-bert/bert-base-uncased",
    "ffn_params": {
        "layers": [768, 384, 7],
        "activation":[nn.ReLU(), nn.ReLU()],
        "dropout": [0.5]
    },
    "optimizers":{
    "adam": {
        "class": torch.optim.Adam,
        "kwargs": {
            "weight_decay": 5e-4,
            "lr": 1e-5,
            }
        }
    },
    "schedulers": {

    }
}