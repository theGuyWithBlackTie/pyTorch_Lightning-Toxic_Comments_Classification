from transformers import AutoModel
from torch import nn


class ToxicCommentTaggerBERT(nn.Module):
    def __init__(self, experiment_params: dict):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(experiment_params["transformer_name"])
        classification_layers = nn.ModuleList()
        for index in range(0, len(experiment_params["ffn_params"]["layers"]) - 2):
            classification_layers.append(nn.Linear(experiment_params["ffn_params"]["layers"][index],
                                                        experiment_params["ffn_params"]["layers"][index + 1]))
            classification_layers.append(experiment_params["ffn_params"]["activation"][index])

            if "dropout" in experiment_params["ffn_params"].keys():
                classification_layers.append(nn.Dropout(experiment_params["ffn_params"]["dropout"][index]))

        classification_layers.append(experiment_params["ffn_params"]["activation"][-1])
        classification_layers.append(
            nn.Linear(experiment_params["ffn_params"]["layers"][-2], experiment_params["ffn_params"]["layers"][-1]))

        self.classification_layers = nn.Sequential(*classification_layers)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output_layer = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        logits = self.classification_layers(output_layer)
        return logits