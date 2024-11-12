import torch
import torch.nn as nn
from transformers import T5Config, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional

class CustomT5EncoderWithMean(T5EncoderModel):
    def __init__(self, config):
        super(CustomT5EncoderWithMean, self).__init__(config)
        self.config = config  # Store the config object
        self.encoder = T5EncoderModel(config)
        self.config.is_encoder_decoder = False
        self.config.decoder_start_token_id = None
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config.added_layers = {
            "final_layer_norm": {"type": "LayerNorm", "d_model": config.d_model},
            "dropout": {"type": "Dropout", "rate": config.dropout_rate}
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state

        # Calculate the mean of the last hidden state along dim=1
        mean_last_hidden_state = last_hidden_state.mean(dim=1)

        # Apply final layer normalization and dropout
        normed_mean_hidden_state = self.final_layer_norm(mean_last_hidden_state)
        output = self.dropout(normed_mean_hidden_state)

        # Wrap the output in BaseModelOutput to include hidden_states
        return BaseModelOutput(last_hidden_state=output, hidden_states=encoder_outputs.hidden_states)
