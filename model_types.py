from transformers import AutoConfig
from packaging import version
import torch.nn as nn
from typing import Optional, Tuple, Union
import torch
import numpy as np

from transformers.modeling_outputs import (
    TokenClassifierOutput,
    SequenceClassifierOutputWithPast, 
    CausalLMOutputWithCrossAttentions)
# if version.parse(transformers.__version__) == version.parse('3.0.2'):
#     from transformers.modeling_gpt2 import GPT2ForSequenceClassification
# else: # transformers: version 4.0
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification
    
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerModel
from transformers import AutoformerConfig, AutoformerForPrediction
import copy
import math

from functools import partial
import modules
import ckconv
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings('once') 


class PlasmaTransformerSeqToLab(GPT2ForSequenceClassification):
    # ~transformers.GPT2ForSequenceClassification` uses the last token in order to do the classification, as
    # other causal models (e.g. GPT-1) do.
    """GPT2ForSequenceClassification with a few modifications."""
    def __init__(
            self,
            n_head,
            n_layer,
            n_inner,
            activation_function,
            attn_pdrop,
            resid_pdrop,
            embd_pdrop,
            layer_norm_epsilon,
            pretrained_model,
            n_embd,
            max_length,
            *args,
            **kwargs):
        # self.config = get_config(kwargs=kwargs)
        transformer_config = AutoConfig.from_pretrained('gpt2')
        transformer_config.n_head = n_head # self.config.num_sent_attn_heads
        transformer_config.n_layer = n_layer # self.config.num_contextual_layers
        transformer_config.n_inner = n_inner
        transformer_config.activation_function = activation_function
        transformer_config.attn_pdrop = attn_pdrop
        transformer_config.n_embd = n_embd # self.config.hidden_dim
        transformer_config.resid_pdrop = resid_pdrop
        # Maximum number of input positions the model can attend to.
        transformer_config.n_positions = max_length # self.config.max_num_sentences + 20 # timestep window to classify over # may need to assign anything above 100 to the 100
        transformer_config.embd_pdrop = embd_pdrop
        # Context window size (should match n_positions).
        transformer_config.n_ctx = transformer_config.n_positions
        transformer_config.pad_token_id = -100
        transformer_config.layer_norm_epsilon = layer_norm_epsilon
        transformer_config.num_labels = 2 # classification task
        
        super().__init__(transformer_config)

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None, # (batch_size, seq_len, hidden_size)
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logits = super().forward(
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True, # just return a plain tuple
        )
        return logits

class ResNetBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_cfg: OmegaConf,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        mask_cfg: OmegaConf,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_cfg.no_hidden
        no_blocks = net_cfg.no_blocks
        data_dim = net_cfg.data_dim
        norm = net_cfg.norm
        dropout = net_cfg.dropout
        dropout_in = net_cfg.dropout_in
        dropout_type = net_cfg.dropout_type
        block_type = net_cfg.block.type
        block_prenorm = net_cfg.block.prenorm
        block_width_factors = net_cfg.block_width_factors
        downsampling = net_cfg.downsampling
        downsampling_size = net_cfg.downsampling_size
        nonlinearity = net_cfg.nonlinearity

        self.data_type = net_cfg.data_type

        # Define dropout_in
        self.dropout_in = torch.nn.Dropout(dropout_in)

        # Unpack conv_type
        conv_type = conv_cfg.type
        # Define partials for types of convs
        ConvType = partial(
            getattr(ckconv.nn, conv_type),
            data_dim=data_dim,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
            mask_cfg=mask_cfg,
        )
        # -------------------------

        # Define NormType
        if norm == "BatchNorm":
            norm_name = f"BatchNorm{data_dim}d"
        else:
            norm_name = norm
        if hasattr(ckconv.nn, norm):
            lib = ckconv.nn
        else:
            lib = torch.nn
        NormType = getattr(lib, norm_name)

        # Define NonlinearType
        NonlinearType = getattr(torch.nn, nonlinearity)

        # Define LinearType
        LinearType = getattr(ckconv.nn, f"Linear{data_dim}d")

        # Define DownsamplingType
        DownsamplingType = getattr(torch.nn, f"MaxPool{data_dim}d")

        # Define Dropout layer type
        DropoutType = getattr(torch.nn, dropout_type)

        # Create Input Layers
        self.conv1 = ConvType(in_channels=in_channels, out_channels=hidden_channels)
        self.norm1 = NormType(hidden_channels)
        self.nonlinear = NonlinearType()

        # Create Blocks
        # -------------------------
        if block_type == "default":
            BlockType = modules.ResNetBlock
        else:
            BlockType = getattr(modules, f"{block_type}Block")
        # 1. Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in ckconv.utils.pairwise_iterable(
                    block_width_factors
                )
            ]
            width_factors = [
                factor for factor_tuple in width_factors for factor in factor_tuple
            ]
        if len(width_factors) != no_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )
        # 2. Create blocks
        blocks = []
        for i in range(no_blocks):
            print(f"Block {i}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                BlockType(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    ConvType=ConvType,
                    NonlinearType=NonlinearType,
                    NormType=NormType,
                    LinearType=LinearType,
                    DropoutType=DropoutType,
                    dropout=dropout,
                    prenorm=block_prenorm,
                )
            )

            # Check whether we need to add a downsampling block here.
            if i in downsampling:
                blocks.append(DownsamplingType(kernel_size=downsampling_size))

        self.blocks = torch.nn.Sequential(*blocks)
        # -------------------------

        # Define Output Layers:
        # -------------------------
        # 1. Calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        # 2. instantiate last layer
        self.out_layer = LinearType(
            in_channels=final_no_hidden, out_channels=out_channels
        )
        # 3. Initialize finallyr
        torch.nn.init.kaiming_normal_(self.out_layer.weight)
        self.out_layer.bias.data.fill_(value=0.0)
        # -------------------------
        if block_type == "S4" and block_prenorm:
            self.out_norm = NormType(final_no_hidden)
        else:
            self.out_norm = torch.nn.Identity()

        # Save variables in self
        self.data_dim = data_dim

    def forward(self, x):
        raise NotImplementedError


class ResNet_sequence(ResNetBase):
    OUTPUT_TYPE = "label"

    # here x is always without lens
    def __blocks_normed(self, x):
        # Dropout in
        x = self.dropout_in(x)
        # First layers
        out = self.nonlinear(self.norm1(self.conv1(x)))
        # Blocks
        out = self.blocks(out)
        # Final layer on last sequence element
        out = self.out_norm(out)
        return out

    def forward(self, x, lens, *args):
        out = self.__blocks_normed(x)
        # Combine masking and multiplying by the denominator for
        # an average of the sequence outputs
        mask = torch.zeros_like(out)
        for i in range(mask.shape[0]):
            mask[i, :, : lens[i]] = (
                1 / lens[i]
            )  # Sets the mask to 1/len for 0...len and 0 otherwise
        out = mask * out
        out = out.sum(dim=-1, keepdim=True)
        # Pass through final projection layer, squeeze & return
        out = self.out_layer(out)
        return out.squeeze(-1)

    def forward_unrolled(self, x, *args):
        out = self.__blocks_normed(x)
        out = torch.cumsum(out, dim=-1)
        out = out / torch.arange(1, out.shape[-1] + 1, device=out.device)
        out = self.out_layer(out)
        return out.squeeze(-2)  # squeeze out channel dim
