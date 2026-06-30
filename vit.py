# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ViT model."""

# copied from https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/vit/modeling_vit.py
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, torch_int
from transformers import ViTConfig as OrgViTConfig
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification as OrgViTForImageClassification,
)
from transformers.models.vit.modeling_vit import (
    ViTPatchEmbeddings,
    ViTAttention,
    ViTIntermediate,
    ViTPooler,
)


logger = logging.get_logger(__name__)


### DiT ###
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaLN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.silu(self.linear(x))


### ViT ###
class ViTConfig(OrgViTConfig):
    def __init__(self, pooling_type: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if pooling_type is None:
            pooling_type = "cls"
        assert pooling_type in ["cls", "mean"], f"Invalid pooling type: {pooling_type}"
        self.pooling_type = pooling_type


class ViTDiTConfig(ViTConfig):
    def __init__(
        self,
        time_conditioning: bool = False,
        cond_hidden_size: int | None = None,
        latent_prediction_head: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.time_conditioning = time_conditioning
        self.latent_prediction_head = latent_prediction_head
        if cond_hidden_size is None:
            self.cond_hidden_size = self.hidden_size // 6
        else:
            self.cond_hidden_size = cond_hidden_size


@dataclass
class BaseModelOutputWithCond(BaseModelOutputWithPooling):
    conditioning: Optional[torch.Tensor] = None


class ViTDiTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTDiTConfig, use_mask_token: bool = False) -> None:
        super().__init__()
        self.time_conditioning = config.time_conditioning
        self.cls_token = None
        if not self.time_conditioning:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            if use_mask_token
            else None
        )
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        num_positions = num_patches + 1
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_positions, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config
        if self.time_conditioning:
            self.label_embeddings = nn.Embedding(config.num_labels, config.hidden_size)

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if (
            not torch.jit.is_tracing()
            and num_patches == num_positions
            and height == width
        ):
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        noisy_embeds: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        if interpolate_pos_encoding:
            raise NotImplementedError("Interpolate pos encoding is not supported")
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        # add the [CLS] token to the embedded patch tokens
        if self.time_conditioning and noisy_embeds is not None:
            cls_tokens = noisy_embeds.unsqueeze(1)
        else:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # if self.time_conditioning and noisy_embeds is not None:
        #     embeddings = torch.cat([embeddings, noisy_embeds.unsqueeze(1)], dim=1)
        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings

        # if self.time_conditioning and noisy_embeds is not None:
        #     embeddings = embeddings + noisy_embeds.unsqueeze(1)

        embeddings = self.dropout(embeddings)

        return embeddings


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTDiTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTDiTConfig) -> None:
        super().__init__()
        self.time_conditioning = config.time_conditioning
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        if config.time_conditioning:
            self.adaLN_modulation = AdaLN(
                config.cond_hidden_size, 6 * config.hidden_size, bias=True
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        residual = hidden_states
        if self.time_conditioning:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(conditioning).chunk(6, dim=1)
            )
        hidden_states = self.layernorm_before(
            hidden_states
        )  # in ViT, layernorm is applied before self-attention
        if self.time_conditioning:
            hidden_states = modulate(hidden_states, shift_msa, scale_msa)
        self_attention_outputs = self.attention(
            hidden_states,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        if self.time_conditioning:
            attention_output = gate_msa.unsqueeze(1) * attention_output
        hidden_states = attention_output + residual

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        if self.time_conditioning:
            layer_output = modulate(layer_output, shift_mlp, scale_mlp)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output)
        if self.time_conditioning:
            layer_output = gate_mlp.unsqueeze(1) * layer_output
        layer_output = layer_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class ViTDiTEncoder(nn.Module):
    def __init__(self, config: ViTDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [ViTDiTLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        layer_indices: Optional[List[int]] = None,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if layer_indices is not None and i not in layer_indices:
                continue
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    conditioning,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, conditioning, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTPreTrainedModel(PreTrainedModel):
    config_class = ViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "ViTEmbeddings",
        "LabelEmbedder",
        "TimestepEmbedder",
        "ViTLayer",
    ]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTDiTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)
            if module.cls_token is not None:
                module.cls_token.data = nn.init.trunc_normal_(
                    module.cls_token.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.cls_token.dtype)

            if module.mask_token is not None:
                module.mask_token.data.zero_()


class ViTForImageClassification(OrgViTForImageClassification):
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.config.pooling_type == "cls":
            sequence_output = sequence_output[:, 0, :]
        elif self.config.pooling_type == "mean":
            sequence_output = sequence_output[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Invalid pooling type: {self.config.pooling_type}")
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ViTDiTModel(ViTPreTrainedModel):
    def __init__(
        self,
        config: ViTDiTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = ViTDiTEmbeddings(config, use_mask_token=use_mask_token)
        if config.time_conditioning:
            self.time_embedder = TimestepEmbedder(config.cond_hidden_size)
        self.encoder = ViTDiTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        if self.config.time_conditioning:
            return self.embeddings.label_embeddings
        else:
            return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noisy_embeds: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        layer_indices: Optional[List[int]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            noisy_embeds=noisy_embeds,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        if self.config.time_conditioning:
            conditioning = F.silu(self.time_embedder(timesteps))
        else:
            conditioning = None

        encoder_outputs = self.encoder(
            embedding_output,
            conditioning=conditioning,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            layer_indices=layer_indices,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:] + (conditioning,)

        return BaseModelOutputWithCond(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            conditioning=conditioning,
        )


class ViTDiTForImageClassification(ViTPreTrainedModel):
    def __init__(self, config: ViTDiTConfig) -> None:
        super().__init__(config)
        self.time_conditioning = config.time_conditioning
        self.num_labels = config.num_labels
        self.vit = ViTDiTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )
        self.latent_delta_head = (
            nn.Linear(config.hidden_size, config.hidden_size)
            if config.latent_prediction_head
            else None
        )
        if config.time_conditioning:
            self.adaLN_modulation = AdaLN(
                config.cond_hidden_size, 2 * config.hidden_size, bias=True
            )

        # Initialize weights and apply final processing
        self.post_init()
        if self.time_conditioning:
            self._init_dit()

    def _init_dit(
        self,
    ):
        # Initialize label embedding table:
        nn.init.normal_(self.vit.embeddings.label_embeddings.weight, std=0.02)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.vit.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.vit.time_embedder.mlp[2].weight, std=0.02)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.vit.encoder.layer:
            nn.init.constant_(block.adaLN_modulation.linear.weight, 0)
            nn.init.constant_(block.adaLN_modulation.linear.bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation.linear.weight, 0)
        nn.init.constant_(self.adaLN_modulation.linear.bias, 0)
        nn.init.constant_(self.classifier.weight, 0)
        nn.init.constant_(self.classifier.bias, 0)
        if self.latent_delta_head is not None:
            nn.init.constant_(self.latent_delta_head.weight, 0)
            nn.init.constant_(self.latent_delta_head.bias, 0)

    def get_input_embeddings(self):
        return self.vit.get_input_embeddings()

    def forward_block(
        self,
        layer_indices: List[int],
        pixel_values: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noisy_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.vit(
            pixel_values,
            timesteps=timesteps,
            noisy_embeds=noisy_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            layer_indices=layer_indices,
        )
        if self.config.pooling_type == "cls":
            outputs.last_hidden_state = outputs[0][:, 0, :]
        elif self.config.pooling_type == "mean":
            outputs.last_hidden_state = outputs[0][:, 1:, :].mean(dim=1)
        else:
            raise ValueError(f"Invalid pooling type: {self.config.pooling_type}")

        # outputs.last_hidden_state = outputs[0][:, 0, :]

        return outputs

    def forward_output_embeddings(
        self, hidden_states: torch.Tensor, conditioning: torch.Tensor
    ):
        if self.config.time_conditioning:
            shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
            hidden_states = modulate(hidden_states, shift, scale)
        logits = self.classifier(hidden_states[:, 0, :])
        return logits

    def forward_latent_delta(
        self, hidden_states: torch.Tensor, conditioning: torch.Tensor
    ):
        if self.latent_delta_head is None:
            raise ValueError("latent_prediction_head is not enabled for this model")
        if self.config.time_conditioning:
            shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
            hidden_states = modulate(hidden_states, shift, scale)
        return self.latent_delta_head(hidden_states[:, 0, :])

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noisy_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.vit(
            pixel_values,
            timesteps=timesteps,
            noisy_embeds=noisy_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        conditioning = outputs[-1]
        if self.config.time_conditioning:
            shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
            sequence_output = modulate(sequence_output, shift, scale)

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_vit(image_size: int, num_labels: int, is_dblock: bool = False, **kwargs):
    if image_size == 32:
        # CIFAR
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_hidden_layers", 12)
        kwargs.setdefault("hidden_size", 128)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("attention_probs_dropout_prob", 0.1)
        kwargs.setdefault("hidden_dropout_prob", 0.1)
    elif image_size == 64:
        # Tiny ImageNet
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_hidden_layers", 12)
        kwargs.setdefault("hidden_size", 768)
        kwargs.setdefault("num_attention_heads", 12)
        kwargs.setdefault("attention_probs_dropout_prob", 0.1)
        kwargs.setdefault("hidden_dropout_prob", 0.1)
    else:
        raise ValueError(f"Invalid image size: {image_size}")
    if is_dblock:
        kwargs["time_conditioning"] = True
        config_cls = ViTDiTConfig
        model_cls = ViTDiTForImageClassification
    else:
        config_cls = ViTConfig
        model_cls = ViTForImageClassification
    config = config_cls(
        image_size=image_size,
        num_labels=num_labels,
        **kwargs,
    )
    model = model_cls(config)
    return model
