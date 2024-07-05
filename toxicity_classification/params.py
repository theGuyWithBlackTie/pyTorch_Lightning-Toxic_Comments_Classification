import torch.nn as nn

experiment_params = {
    "tokenizer_type": "google-bert/bert-base-uncased",
    "transformer_name": "google-bert/bert-base-uncased",
    "ffn_params": {
        "layers": [768, 384, 7],
        "activation":[nn.ReLU(), nn.ReLU()],
        "dropout": [0.5]
    },
}