#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang
#                                                  Mingshuang Luo)
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


import argparse
import logging
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformer import Noam

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    encode_supervisions,
    setup_logger,
    str2bool,
)
from icefall.env import get_env_info
import copy
import collections
import math
from quantization import Quantizer
import random


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--bytes-per-frame",
        type=int,
        default=4,
        help="number of code books",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--predictor",
        type=str,
        default="ckpnt_predictor",
        help="simple_linear predictor ckpnt_predictor",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        conformer_ctc/exp/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp_finetune_joint_quantizer",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_1000",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--att-rate",
        type=float,
        default=0.0,
        help="""The attention rate.
        The total loss is (1 -  att_rate) * ctc_loss + att_rate * att_loss
        """,
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        # default=None,
        default="conformer_ctc/exp/model.pt",
        help="use a pretrained model, e.g. a modle downloaded from model zoo",
    )

    parser.add_argument(
        "--use-quantizer",
        type=str2bool,
        default=True,
        help="Should quantizer be used.",
    )

    parser.add_argument(
        "--quantizer-weight",
        type=float,
        default=1.0,
        help="""Weight of the quantizer loss"
        """,
    )

    parser.add_argument(
        "--entropy-scale",
        type=float,
        default=0.1,
        help="""Sclae of the logits_entropy_loss"
        """,
    )

    parser.add_argument(
        "--pretrained-quantizer",
        type=str,
        default=None,
        # default="conformer_ctc/exp/mem/022c7e67-bytes_per_frame_4_utt_1k-quantizer.pt",
        help="use a pretrained quantizer model",
    )

    parser.add_argument(
        "--memory-embedding-dim",
        type=int,
        default=256,
        help="dim of memory embeddings to train quantizer"
    )

    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=1.0,
        help="""Weight of the consistency loss"
        """,
    )

    parser.add_argument(
        "--consistency-level",
        type=str,
        default="feature",
        help="""Level of the consistency loss"
        """,
    )

    parser.add_argument(
        "--pseudo-label",
        type=str2bool,
        default=True,
        help="Should pseudo labels be generated.",
    )

    parser.add_argument(
        "--ema-decay-factor",
        type=float,
        default=0.99989,
        help="""decay factor for exponential moving average of the teacher model (0.0 is always use current model)"
        """,
    )

    parser.add_argument(
        "--optimizer-type",
        type=str,
        default="adam",
        help="""Optimizer (adam or noam)"
        """,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="""Learning rate for Adam optimizer"
        """,
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - use_feat_batchnorm: Whether to do batch normalization for the
                              input features.

        - attention_dim: Hidden dim for multi-head attention model.

        - head: Number of heads of multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss

    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 1000,
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "use_feat_batchnorm": True,
            "attention_dim": 256,
            "nhead": 4,
            "num_decoder_layers": 0,
            # parameters for loss
            "beam_size": 10,
            "reduction": "sum",
            "use_double_scores": True,
            # parameters for Noam
            "weight_decay": 1e-6,
            "lr_factor": 5.0,
            "warm_step": 25000,
            "env_info": get_env_info(),
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Load checkpoint from file.

    If params.start_epoch is positive, it will load the checkpoint from
    `params.start_epoch - 1`. Otherwise, this function does nothing.

    Apart from loading state dict for `model`, `optimizer` and `scheduler`,
    it also updates `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The learning rate scheduler we are using.
    Returns:
      Return None.
    """
    if params.start_epoch <= 0:
        return

    filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    teacher_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
    )
    if teacher_model is not None:
        save_checkpoint_impl(
            filename=params.exp_dir / f"teacher-epoch-{params.cur_epoch}.pt",
            model=teacher_model,
            rank=rank,
        )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def pseudo_label(lprobs, supervision_segments):
    ctc_targets = list()
    indices = supervision_segments[:, 0]
    input_lengths = supervision_segments[:, 2]
    for i, indice in enumerate(indices):
        lprob = lprobs[indice][:input_lengths[i]]
        _, max_toks = lprob.max(dim=-1)
        toks = max_toks.unique_consecutive()
        pred_units_arr = toks[toks != 0]
        ctc_targets.append(pred_units_arr.tolist())

    return ctc_targets


def pseudo_label_sample(lprobs, supervision_segments):
    ctc_targets = list()
    indices = supervision_segments[:, 0]
    input_lengths = supervision_segments[:, 2]
    for i, indice in enumerate(indices):
        prob = lprobs[indice][:input_lengths[i]].exp()
        sampler = torch.distributions.categorical.Categorical(prob)
        max_toks = sampler.sample()
        toks = max_toks.unique_consecutive()
        pred_units_arr = toks[toks != 0]
        ctc_targets.append(pred_units_arr.tolist())

    return ctc_targets


def SoftNLLLoss(
    log_probs, targets, mask
):
    """
    negative log likelihood loss using soft labels.

    Args:
    log_probs : The predicted log-probs. Its shape is [batch, frames, p]
    targets : The target probabilities. Its shape is [batch, frames, p]
    mask : Padding mask, where the padding indices are True. Its shape is [batch, frames]
    """
    # Getting the number of sentences in the minibatch
    batch_size = log_probs.shape[0]

    # Getting the maximum length of label sequence
    max_len = log_probs.shape[1]

    # Reshape to [batch_size * length, feature]
    log_probs = log_probs.reshape(batch_size * max_len, log_probs.shape[-1])

    # Reshape to [batch_size * length, feature]
    targets = targets.reshape(batch_size * max_len, targets.shape[-1])

    loss = (-targets * log_probs).sum(1)
    # Loss averaging
    loss = torch.sum(loss.reshape(batch_size, max_len) * ~mask) 
    return loss


def EMA(new_model, last_model, decay_factor):
    model_params_keys = list(new_model.keys())
    params_keys = list(last_model.keys())
    if params_keys != model_params_keys:
        raise KeyError(
            "expected list of params: {}, "
            "but found: {}".format(params_keys, model_params_keys)
        )

    latest_model = collections.OrderedDict()

    for k in params_keys:
        p_new = new_model[k].float()
        p_last = last_model[k].float()
        latest_model[k] = decay_factor * p_last + (1 - decay_factor) * p_new
    return latest_model


def compute_semi_loss(
    params: AttributeDict,
    model: nn.Module,
    teacher_model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute CTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      graph_compiler:
        It is used to build a decoding graph from a ctc topo and training
        transcript. The training transcript is contained in the given `batch`,
        while the ctc topo is built when this compiler is instantiated.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = graph_compiler.device

    supervisions = batch["supervisions"]

    triple_feature = [batch["inputs_raw"], batch["inputs"], batch["inputs_copy"]]
    assert len(triple_feature) == 3

    # each feature is (N, T, C)
    assert triple_feature[0].ndim == 3

    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    # identify labeled and unlabeled ids
    all_token_ids = graph_compiler.texts_to_ids(texts)
    label_token_ids = []
    unlabel_text_ids = []
    label_text_ids = []
    for i in range(len(all_token_ids)):
        if len(all_token_ids[i]) == 0:
            unlabel_text_ids.append(i)
        else:
            label_text_ids.append(i)
            label_token_ids.append(all_token_ids[i])
    if len(unlabel_text_ids) > 0:
        unlabel_text_ids = torch.tensor(unlabel_text_ids)
        unlabel_audio_ids = supervision_segments[unlabel_text_ids][:, 0].long()
    else:
        unlabel_audio_ids = None

    if teacher_model is not None:
        teacher_model_state_dict = EMA(model.state_dict(), teacher_model.state_dict(), params.ema_decay_factor)
        teacher_model.load_state_dict(teacher_model_state_dict)
        teacher_model.eval()

        with torch.no_grad():
            nnet_output_infer,feature_infer, feature_mask_infer = teacher_model(triple_feature[0].to(device), supervisions)
            pseudo_token_ids = pseudo_label(nnet_output_infer, supervision_segments)
        for i in range(len(all_token_ids)):
            if len(all_token_ids[i]) == 0:
                all_token_ids[i] = pseudo_token_ids[i]
        if params.consistency_level == "feature":
            all_feature_infer = feature_infer.transpose(0,1)[~feature_mask_infer]
            # take unlabeled part of features
            if unlabel_audio_ids != None:
                feature_infer = feature_infer.transpose(0,1)[unlabel_audio_ids][~feature_mask_infer[unlabel_audio_ids]]
            else:
                feature_infer = torch.tensor([], device=feature_infer.device)

    model.train()
    nnet_output_1, feature_1, feature_mask_1 = model(triple_feature[1].to(device), supervisions)
    if params.consistency_level == "feature":
        all_feature_1 = feature_1.transpose(0,1)[~feature_mask_1]
        # take unlabeled part of features
        if unlabel_audio_ids != None:
            feature_1 = feature_1.transpose(0,1)[unlabel_audio_ids][~feature_mask_1[unlabel_audio_ids]]
        else:
            feature_1 = torch.tensor([], device=feature_1.device)

    mmodel = model.module if hasattr(model, "module") else model
    nnet_output_2, feature_2, feature_mask_2 = mmodel(triple_feature[2].to(device), supervisions)
    if params.consistency_level == "feature":
        all_feature_2 = feature_2.transpose(0,1)[~feature_mask_2]
        # take unlabeled part of features
        if unlabel_audio_ids != None:
            feature_2 = feature_2.transpose(0,1)[unlabel_audio_ids][~feature_mask_2[unlabel_audio_ids]]
        else:
            feature_2 = torch.tensor([], device=feature_2.device)

    assert feature_mask_1.equal(feature_mask_2)

    # compute ctc loss for both labeled and unlabeled part
    decoding_graph = graph_compiler.compile(all_token_ids)

    dense_fsa_vec_1 = k2.DenseFsaVec(
        nnet_output_1,
        supervision_segments,
        allow_truncate=params.subsampling_factor - 1,
    )
    dense_fsa_vec_2 = k2.DenseFsaVec(
        nnet_output_2,
        supervision_segments,
        allow_truncate=params.subsampling_factor - 1,
    )

    ctc_loss = 1/2 * ( k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec_1,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    ) + k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec_2,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    ))

    num_frames = supervision_segments[:, 2].sum().item()

    if params.use_quantizer:
        mmodel = model.module if hasattr(model, "module") else model
        num_iters = 2 if random.random() < 0.5 else 1
        (reconstruction_loss, logprob_loss,
         logits_entropy_loss, index_entropy_loss) = mmodel.quantizer.compute_loss(all_feature_infer.detach(), num_iters)
        quantizer_loss = (reconstruction_loss +
                    logprob_loss +
                    logits_entropy_loss * params.entropy_scale)
    
        quantizer_loss = quantizer_loss * num_frames
    else:
        quantizer_loss = torch.zeros((1), device=nnet_output_1.device)

    # compute consistency loss for the unlabeled part
    if params.consistency_weight != 0.0 and feature_infer.size(0) > 0:
        if params.consistency_level == "logit":
            consistency_loss = 1/2 * (SoftNLLLoss(nnet_output_1, nnet_output_2.exp(), feature_mask_1) + SoftNLLLoss(nnet_output_2, nnet_output_1.exp(), feature_mask_1)) 
        elif params.consistency_level == "feature":
            # if use quantizer, compute the codebook loss
            if params.use_quantizer:
                encoded_feature_infer = mmodel.quantizer.encode(feature_infer)
                consistency_loss = 1/2 * ( mmodel.cdidxnet(
                        feature_1, encoded_feature_infer
                    ) + mmodel.cdidxnet(
                        feature_2, encoded_feature_infer
                    ))
            # if not, compute the mse loss
            else:
                consistency_loss = 1/2 * ( F.mse_loss(feature_infer.detach(), feature_1, reduction='sum')
                    + F.mse_loss(feature_infer.detach(), feature_2, reduction='sum'))
        else:
            raise NotImplementedError()
    else:
        # compute the codebook loss even if no unlabeled data is available in this batch
        # to avoid the error "not all parameters were used"
        if params.use_quantizer and params.consistency_level == "feature":
            encoded_feature_infer = mmodel.quantizer.encode(all_feature_infer)
            consistency_loss = 0.0 * ( mmodel.cdidxnet(
                    all_feature_1, encoded_feature_infer
                ) + mmodel.cdidxnet(
                    all_feature_2, encoded_feature_infer
                ))
        else:
            consistency_loss = torch.zeros((1), device=nnet_output_1.device)

    assert params.att_rate == 0.0

    loss = ctc_loss + params.consistency_weight * consistency_loss + params.quantizer_weight * quantizer_loss
    assert loss.requires_grad == is_training

    info = MetricsTracker()
    info["frames"] = num_frames
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    info["consistency_loss"] = consistency_loss.detach().cpu().item()
    if params.use_quantizer:
        info["quantizer_loss"] = quantizer_loss.detach().cpu().item()
        info["reconstruction_loss"] = reconstruction_loss.detach().cpu().item() * num_frames
        info["logprob_loss"] = logprob_loss.detach().cpu().item() * num_frames
        info["logits_entropy_loss"] = logits_entropy_loss.detach().cpu().item() * num_frames
        info["index_entropy_loss"] = index_entropy_loss.detach().cpu().item() * num_frames


    info["loss"] = loss.detach().cpu().item()

    return loss, info


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute CTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      graph_compiler:
        It is used to build a decoding graph from a ctc topo and training
        transcript. The training transcript is contained in the given `batch`,
        while the ctc topo is built when this compiler is instantiated.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = graph_compiler.device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    with torch.set_grad_enabled(is_training):
        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
        # nnet_output is (N, T, C)

    # NOTE: We need `encode_supervisions` to sort sequences with
    # different duration in decreasing order, required by
    # `k2.intersect_dense` called in `k2.ctc_loss`
    supervision_segments, texts = encode_supervisions(
        supervisions, subsampling_factor=params.subsampling_factor
    )

    token_ids = graph_compiler.texts_to_ids(texts)

    decoding_graph = graph_compiler.compile(token_ids)

    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=params.subsampling_factor - 1,
    )

    ctc_loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=params.beam_size,
        reduction=params.reduction,
        use_double_scores=params.use_double_scores,
    )

    if params.att_rate != 0.0:
        with torch.set_grad_enabled(is_training):
            mmodel = model.module if hasattr(model, "module") else model
            # Note: We need to generate an unsorted version of token_ids
            # `encode_supervisions()` called above sorts text, but
            # encoder_memory and memory_mask are not sorted, so we
            # use an unsorted version `supervisions["text"]` to regenerate
            # the token_ids
            #
            # See https://github.com/k2-fsa/icefall/issues/97
            # for more details
            unsorted_token_ids = graph_compiler.texts_to_ids(
                supervisions["text"]
            )
            att_loss = mmodel.decoder_forward(
                encoder_memory,
                memory_mask,
                token_ids=unsorted_token_ids,
                sos_id=graph_compiler.sos_id,
                eos_id=graph_compiler.eos_id,
            )
        loss = (1.0 - params.att_rate) * ctc_loss + params.att_rate * att_loss
    else:
        loss = ctc_loss
        att_loss = torch.tensor([0])

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    info["frames"] = supervision_segments[:, 2].sum().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    if params.att_rate != 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()

    info["loss"] = loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    teacher_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      graph_compiler:
        It is used to convert transcripts to FSAs.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        loss, loss_info = compute_semi_loss(
            params=params,
            model=model,
            teacher_model=teacher_model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=True,
        )
        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}"
            )

        if batch_idx % params.log_interval == 0:

            if tb_writer is not None:
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(
                    tb_writer, "train/tot_", params.batch_idx_train
                )

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(42)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    graph_compiler = BpeCtcTrainingGraphCompiler(
        params.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )

    logging.info("About to create model")
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=False,
        use_feat_batchnorm=params.use_feat_batchnorm,
        use_codebook_loss=True if params.use_quantizer else False,
        num_codebooks=params.bytes_per_frame,
        predictor=params.predictor,
    )

    if params.pretrained_model is not None:
        load_checkpoint(params.pretrained_model, model=model)

    if params.use_quantizer:
        quantizer = Quantizer(dim=params.memory_embedding_dim, num_codebooks=args.bytes_per_frame, codebook_size=256)
        if params.pretrained_quantizer is not None:
            quantizer.load_state_dict(torch.load(params.pretrained_quantizer))
        model.add_module("quantizer", quantizer)

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
    if params.pseudo_label:
        teacher_model = copy.deepcopy(model)
    else:
        teacher_model = None

    if params.optimizer_type == "noam":
        optimizer = Noam(
            model.parameters(),
            model_size=params.attention_dim,
            factor=params.lr_factor,
            warm_step=params.warm_step,
            weight_decay=params.weight_decay,
        )
    elif params.optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params.lr, betas=(0.9, 0.98), eps=1e-9
        )
    else:
        raise NotImplementedError()

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"], strict=False)

    librispeech = LibriSpeechAsrDataModule(args)
    train_dl = librispeech.train_semi_dataloaders()
    valid_dl = librispeech.valid_dataloaders()

    scan_pessimistic_batches_for_oom(
        model=model,
        teacher_model=teacher_model,
        train_dl=train_dl,
        optimizer=optimizer,
        graph_compiler=graph_compiler,
        params=params,
    )

    for epoch in range(params.start_epoch, params.num_epochs):
        train_dl.sampler.set_epoch(epoch)

        if params.optimizer_type == "noam":
            cur_lr = optimizer._rate
        elif params.optimizer_type == "adam":
                cur_lr = optimizer.param_groups[0]['lr']
        else:
            raise NotImplementedError()
        if tb_writer is not None:
            tb_writer.add_scalar(
                "train/learning_rate", cur_lr, params.batch_idx_train
            )
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        if rank == 0:
            logging.info("epoch {}, learning rate {}".format(epoch, cur_lr))

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            teacher_model=teacher_model,
            optimizer=optimizer,
            graph_compiler=graph_compiler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
        )

        save_checkpoint(
            params=params,
            model=model,
            teacher_model=teacher_model,
            optimizer=optimizer,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def scan_pessimistic_batches_for_oom(
    model: nn.Module,
    teacher_model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 0 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            optimizer.zero_grad()
            loss, _ = compute_semi_loss(
                params=params,
                model=model,
                teacher_model=teacher_model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=True,
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0, 2.0)
            optimizer.step()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
