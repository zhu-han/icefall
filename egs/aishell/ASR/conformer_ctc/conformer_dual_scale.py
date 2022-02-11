#!/usr/bin/env python3
# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


import math
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformer import Supervisions, Transformer, encoder_padding_mask
from conformer import RelPositionMultiheadAttention, Swish, identity, RelPositionalEncoding


class Conformer(Transformer):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        subsampling_factor (int): subsampling factor of encoder (the convolution layers before transformers)
        d_model (int): attention dimension
        nhead (int): number of head
        dim_feedforward (int): feedforward dimention
        num_encoder_layers (int): number of encoder layers
        num_decoder_layers (int): number of decoder layers
        dropout (float): dropout rate
        cnn_module_kernel (int): Kernel size of convolution module
        normalize_before (bool): whether to use layer_norm before the first block.
        vgg_frontend (bool): whether to use vgg frontend.
        use_feat_batchnorm(bool): whether to use batch-normalize the input.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        sub_d_model: int = 256,
        nhead: int = 4,
        sub_nhead: int = 4,
        dim_feedforward: int = 2048,
        sub_dim_feedforward: int=2048,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
        vgg_frontend: bool = False,
        use_feat_batchnorm: bool = False,
    ) -> None:
        super(Conformer, self).__init__(
            num_features=num_features,
            num_classes=num_classes,
            subsampling_factor=subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            normalize_before=normalize_before,
            vgg_frontend=vgg_frontend,
            use_feat_batchnorm=use_feat_batchnorm,
        )

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)
        self.sub_encoder_pos = RelPositionalEncoding(sub_d_model, dropout)

        encoder_layer = ConformerEncoderLayer(
            d_model,
            sub_d_model,
            nhead,
            sub_nhead,
            dim_feedforward,
            sub_dim_feedforward,
            dropout,
            cnn_module_kernel,
            normalize_before,
        )
        self.encoder = ConformerEncoder(encoder_layer, num_encoder_layers)
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(d_model)
            self.sub_after_norm = nn.LayerNorm(sub_d_model)
        else:
            # Note: TorchScript detects that self.after_norm could be used inside forward()
            #       and throws an error without this change.
            self.after_norm = identity


    def forward(
        self, x: torch.Tensor, supervision: Optional[Supervisions] = None, sub: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (N, T, C).
          supervision:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            (CAUTION: It contains length information, i.e., start and number of
             frames, before subsampling)

        Returns:
          Return a tuple containing 3 tensors:
            - CTC output for ctc decoding. Its shape is (N, T, C)
            - Encoder output with shape (T, N, C). It can be used as key and
              value for the decoder.
            - Encoder output padding mask. It can be used as
              memory_key_padding_mask for the decoder. Its shape is (N, T).
              It is None if `supervision` is None.
        """
        if self.use_feat_batchnorm:
            x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
            x = self.feat_batchnorm(x)
            x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        encoder_memory, memory_key_padding_mask = self.run_encoder(
            x, supervision, sub=sub
        )
        x = self.ctc_output(encoder_memory)
        return x, encoder_memory, memory_key_padding_mask

    def run_encoder(
        self, x: Tensor, supervisions: Optional[Supervisions] = None, sub: Optional[bool] = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
          x:
            The model input. Its shape is [N, T, C].
          supervisions:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute encoder padding mask, which is used as memory key padding
            mask for the decoder.

        Returns:
            Tensor: Predictor tensor of dimension (input_length, batch_size, d_model).
            Tensor: Mask tensor of dimension (batch_size, input_length)
        """
        x = self.encoder_embed(x)
        if not sub:
            x, pos_emb = self.encoder_pos(x)
        else:
            x, pos_emb = self.sub_encoder_pos(x)
        x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        mask = encoder_padding_mask(x.size(0), supervisions)
        if mask is not None:
            mask = mask.to(x.device)
        x = self.encoder(x, pos_emb, src_key_padding_mask=mask, sub=sub)  # (T, B, F)

        if self.normalize_before:
            if not sub:
                x = self.after_norm(x)
            else:
                x = self.sub_after_norm(x)

        return x, mask


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.
        normalize_before: whether to use layer_norm before the first block.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        sub_d_model: int,
        nhead: int,
        sub_nhead: int,
        dim_feedforward: int = 2048,
        sub_dim_feedforward: int = 1024,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = SubRelPositionMultiheadAttention(
            d_model, sub_d_model, nhead, sub_nhead, dropout=0.0
        )

        self.feed_forward_1 = SubLinear(d_model, dim_feedforward, sub_d_model, sub_dim_feedforward)
        self.feed_forward_2 = nn.Sequential(
            Swish(),
            nn.Dropout(dropout)
        )
        self.feed_forward_3 = SubLinear(dim_feedforward, d_model, sub_dim_feedforward, sub_d_model)

        self.feed_forward_macaron_1 = SubLinear(d_model, dim_feedforward, sub_d_model, sub_dim_feedforward)
        self.feed_forward_macaron_2 = nn.Sequential(
            Swish(),
            nn.Dropout(dropout)
        )
        self.feed_forward_macaron_3 = SubLinear(dim_feedforward, d_model, sub_dim_feedforward, sub_d_model)

        self.conv_module = SubConvolutionModule(d_model, sub_d_model, cnn_module_kernel)

        self.norm_ff_macaron = nn.LayerNorm(
            d_model
        )  # for the macaron style FNN module
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module
        self.norm_mha = nn.LayerNorm(d_model)  # for the MHA module

        self.ff_scale = 0.5

        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(
            d_model
        )  # for the final output of the block

        self.sub_norm_ff_macaron = nn.LayerNorm(
            sub_d_model
        )  # for the macaron style FNN module
        self.sub_norm_ff = nn.LayerNorm(sub_d_model)  # for the FNN module
        self.sub_norm_mha = nn.LayerNorm(sub_d_model)  # for the MHA module

        self.sub_norm_conv = nn.LayerNorm(sub_d_model)  # for the CNN module
        self.sub_norm_final = nn.LayerNorm(
            sub_d_model
        )  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        sub: Optional[Tensor] = False,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """

        # macaron style feed forward module
        residual = src
        if self.normalize_before:
            if not sub:
                src = self.norm_ff_macaron(src)
            else:
                src = self.sub_norm_ff_macaron(src)
        src = self.feed_forward_macaron_2(self.feed_forward_macaron_1(src, sub))
        src = self.feed_forward_macaron_3(src, sub)
        src = residual + self.ff_scale * self.dropout(src)
        if not self.normalize_before:
            if not sub:
                src = self.norm_ff_macaron(src)
            else:
                src = self.sub_norm_ff_macaron(src)

        # multi-headed self-attention module
        residual = src
        if self.normalize_before:
            if not sub:
                src = self.norm_mha(src)
            else:
                src = self.sub_norm_mha(src)
        src_att = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            sub=sub,
        )[0]
        src = residual + self.dropout(src_att)
        if not self.normalize_before:
            if not sub:
                src = self.norm_mha(src)
            else:
                src = self.sub_norm_mha(src)

        # convolution module
        residual = src
        if self.normalize_before:
            if not sub:
                src = self.norm_conv(src)
            else:
                src = self.sub_norm_conv(src)
        src = residual + self.dropout(self.conv_module(src))
        if not self.normalize_before:
            if not sub:
                src = self.norm_conv(src)
            else:
                src = self.sub_norm_conv(src)

        # feed forward module
        residual = src
        if self.normalize_before:
            if not sub:
                src = self.norm_ff(src)
            else:
                src = self.sub_norm_ff(src)
        src = self.feed_forward_2(self.feed_forward_1(src, sub))
        src = self.feed_forward_3(src, sub)
        src = residual + self.ff_scale * self.dropout(src)
        if not self.normalize_before:
            if not sub:
                src = self.norm_ff(src)
            else:
                src = self.sub_norm_ff(src)

        if self.normalize_before:
            if not sub:
                src = self.norm_final(src)
            else:
                src = self.sub_norm_final(src)

        return src


class ConformerEncoder(nn.TransformerEncoder):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None
    ) -> None:
        super(ConformerEncoder, self).__init__(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=norm
        )

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        sub: Optional[Tensor] = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        output = src

        for i, mod in enumerate(self.layers):
            if i % 2 == 0 or not sub:
                output = mod(
                    output,
                    pos_emb,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    sub=sub,
                )

        if self.norm is not None:
            output = self.norm(output)

        return output


class SubRelPositionMultiheadAttention(RelPositionMultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        sub_embed_dim: int,
        num_heads: int,
        sub_num_heads: int,  
        dropout: float = 0.0,
    ) -> None:
        super(SubRelPositionMultiheadAttention, self).__init__(embed_dim, num_heads, dropout)
        self.sub_embed_dim = sub_embed_dim
        self.sub_num_heads = sub_num_heads
        self.sub_head_dim = sub_embed_dim // num_heads

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        sub: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not sub:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                pos_emb,
                self.embed_dim,
                self.num_heads,
                self.in_proj.weight,
                self.in_proj.bias,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.linear_pos.weight,
                self.pos_bias_u,
                self.pos_bias_v,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                pos_emb,
                self.sub_embed_dim,
                self.sub_num_heads,
                self.in_proj.weight[:3 * self.sub_embed_dim, :self.sub_embed_dim],
                self.in_proj.bias[:self.sub_embed_dim],
                self.dropout,
                self.out_proj.weight[:self.sub_embed_dim, :self.sub_embed_dim],
                self.out_proj.bias[:self.sub_embed_dim],
                self.linear_pos.weight[:self.sub_embed_dim, :self.sub_embed_dim],
                self.pos_bias_u[:self.sub_num_heads, :self.sub_head_dim],
                self.pos_bias_v[:self.sub_num_heads, :self.sub_head_dim],
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        linear_pos_weight: Tensor,
        pos_bias_u: Tensor,
        pos_bias_v: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` or :math:`(1, 2*L-1, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = nn.functional.linear(
                query, in_proj_weight, in_proj_bias
            ).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError(
                        "The size of the 2D attn_mask is not correct."
                    )
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError(
                        "The size of the 3D attn_mask is not correct."
                    )
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()
                    )
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if (
            key_padding_mask is not None
            and key_padding_mask.dtype == torch.uint8
        ):
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == src_len, "{} == {}".format(
                key_padding_mask.size(1), src_len
            )

        q = q.transpose(0, 1)  # (batch, time1, head, d_k)

        pos_emb_bsz = pos_emb.size(0)
        assert pos_emb_bsz in (1, bsz)  # actually it is 1
        p = nn.functional.linear(pos_emb, linear_pos_weight).view(pos_emb_bsz, -1, num_heads, head_dim)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        q_with_bias_u = (q + pos_bias_u).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        q_with_bias_v = (q + pos_bias_v).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)
        matrix_ac = torch.matmul(
            q_with_bias_u, k
        )  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(
            q_with_bias_v, p.transpose(-2, -1)
        )  # (batch, head, time1, 2*time1-1)
        matrix_bd = self.rel_shift(matrix_bd)

        attn_output_weights = (
            matrix_ac + matrix_bd
        ) * scaling  # (batch, head, time1, time2)

        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, -1
        )

        assert list(attn_output_weights.size()) == [
            bsz * num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim)
        )
        attn_output = nn.functional.linear(
            attn_output, out_proj_weight, out_proj_bias
        )

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None


class SubConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self, channels: int, sub_channels: int, kernel_size: int, bias: bool = True
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(SubConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = SubConv1d(
            channels,
            2 * channels,
            sub_channels,
            2 * sub_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = SubConv1d(
            channels,
            channels,
            sub_channels,
            sub_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.subnorm = nn.BatchNorm1d(sub_channels)
        self.pointwise_conv2 = SubConv1d(
            channels,
            channels,
            sub_channels,
            sub_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(self, x: Tensor, sub: bool) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x, sub)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x, sub)
        if sub:
            x = self.subnorm(x)
        else:
            x = self.norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x, sub)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class SubLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, sub_in_features: int,  sub_out_features: int,  bias: bool = True,
                 device=None, dtype=None) -> None:
        super(SubLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.sub_in_features = sub_in_features
        self.sub_out_features = sub_out_features

    def forward(self, input: Tensor, sub: bool) -> Tensor:
        if not sub:
            return nn.functional.linear(input, self.weight, self.bias)
        else:
            return nn.functional.linear(input, self.weight[:self.sub_out_features, :self.sub_in_features], self.bias[:self.sub_out_features])

from torch.nn.common_types  import _size_1_t

class SubConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sub_in_channels: int,
        sub_out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(SubConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.sub_in_channels = sub_in_channels
        self.sub_out_channels = sub_out_channels

    def forward(self, input: Tensor, sub: bool) -> Tensor:
        if not sub:
            return self._conv_forward(input, self.weight, self.bias)
        else:
            return self._conv_forward(input, self.weight[:self.sub_out_channels, :self.sub_in_channels // self.groups], self.bias[:self.sub_out_channels])

