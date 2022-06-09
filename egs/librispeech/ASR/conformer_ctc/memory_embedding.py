#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (author: Liyong Guo)
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
from enum import Flag
import logging
from pathlib import Path
from typing import List, Tuple

import k2
import torch
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from lhotse.features.io import FeaturesWriter, NumpyHdf5Writer
from lhotse import CutSet

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.decode import one_best_decoding
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    encode_supervisions,
    get_alignments,
    save_alignments,
    setup_logger,
)
import math


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_1000",
        help="The lang dir",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--mem-dir",
        type=str,
        default="conformer_ctc/exp/mem",
        help="The experiment dir",
    )

    parser.add_argument(
        "--num-utts",
        type=int,
        default=1000,
        help="number of utts to extract memory embeddings",
    )

    parser.add_argument(
        "--pretrained_model",
        type=Path,
        default="conformer_ctc/exp/model.pt",
        help="use a pretrained model, e.g. a modle downloaded from model zoo",
    )
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "feature_dim": 80,
            "nhead": 4,
            "attention_dim": 256,
            "subsampling_factor": 4,
            "num_decoder_layers": 6,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            "output_beam": 10,
            "use_double_scores": True,
            "env_info": get_env_info(),
        }
    )
    return params

def compute_memory(
    model: torch.nn.Module,
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    writer: None,
) -> List[Tuple[str, List[int]]]:
    """Compute the framewise alignments of a dataset.

    Args:
      model:
        The neural network model.
      dl:
        Dataloader containing the dataset.
      params:
        Parameters for computing memory.
    Returns:
      Return a list of tuples. Each tuple contains two entries:
        - Utterance ID
        - memory embeddings
    """
    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"
    num_cuts = 0

    device = params.device
    cuts = []
    total_frames = 0
    done_flag = False
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            feature = batch["inputs"]

            # at entry, feature is [N, T, C]
            assert feature.ndim == 3
            feature = feature.to(device)

            supervisions = batch["supervisions"]

            _, encoder_memory, memory_mask = model(feature, supervisions)

            # [T, N, C] --> [N, T, C]
            encoder_memory = encoder_memory.transpose(0, 1).to("cpu").numpy()

            cut_list = supervisions["cut"]
            assert len(cut_list) == encoder_memory.shape[0]
            assert all(supervisions["start_frame"] == 0)
            num_cuts += len(cut_list)
            for idx, cut in enumerate(cut_list):
                num_frames = (~memory_mask)[idx].sum().item()
                cut.encoder_memory = writer.store_array(
                    key=cut.id,
                    value=encoder_memory[idx][:num_frames],
                )
                total_frames += num_frames

            cuts += cut_list
            if len(cuts) > params.num_utts:
                break
            if batch_idx % 100 == 0:
                logging.info(f"processed {total_frames} frames and {num_cuts} cuts; {batch_idx} of {num_batches}")
    return CutSet.from_cuts(cuts)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    assert args.return_cuts is True
    assert args.concatenate_cuts is False

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/mem")

    logging.info("Computing memory embedings- started")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    params["device"] = device

    logging.info("About to create model")
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=params.vgg_frontend,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )
    if params.pretrained_model is not None:
        load_checkpoint(params.pretrained_model, model=model)

    model.to(device)
    model.eval()

    args.enable_augmentation = False
    librispeech = LibriSpeechAsrDataModule(args)

    train_dl = librispeech.train_dataloaders()

    mem_dir = Path(params.mem_dir)
    mem_dir.mkdir(exist_ok=True)

    with NumpyHdf5Writer(mem_dir / "memory_embeddings_1k") as writer:
        cut_set = compute_memory(
            model=model,
            dl=train_dl,
            params=params,
            writer=writer,
        )
        cut_set.to_json(mem_dir / "memory_manifest_1k.json")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
