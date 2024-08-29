
# Introduction

This recipe includes scripts for training LLM-based ASR models, where speech features extracted from a speech encoder are fed to the LLM, and the LLM generates the corresponding transcriptions.

[./RESULTS.md](./RESULTS.md) contains the latest results.

The following table lists the folders for different tasks.

|     Recipe   |  Speech Encoder             | LLM   |  Language          | Fine-tuning Dataset                                           |
|---------| ------------------|---------------------|--------------------|---------------------------------------------------|
| [zipformer_llm_en](./zipformer_llm_en)   |  Zipformer           | Qwen2   |     English         |  LibriSpeech |
| [whisper_llm_zh](./whisper_llm_zh)        |  Whisper           | Qwen2    |    Chinese          | Aishell1 or [Multiple Chinese datasets](https://github.com/k2-fsa/icefall/tree/master/egs/multi_zh-hans/ASR)                                                 |


# How to use

## 1. zipformer_llm_en

### 1.1. Install dependencies

```bash
pip install -r zipformer_llm_en/requirements.txt
pip install huggingface_hub['cli']
```

### 1.2. Prepare data

```bash
zipformer_llm_en/prepare.sh
```

This script following the `prepare.sh` in the LibriSpeech recipe to prepare the data.
Your can skip this step if you have already prepared the data.

### 1.3. Prepare pre-trained speech encoder and LLM model

Download the Qwen2-1.5B-Instruct LLM model.
```bash
mkdir -p models/qwen2-1.5b-instruct
huggingface-cli download  --local-dir models/qwen2-1.5b-instruct     Qwen/Qwen2-1.5B-Instruct
```

For baseline model, download one pre-trained Zipformer model:

```bash
mkdir -p models/zipformer/librispeech/transducer_crctc_large

wget -P models/zipformer/librispeech/transducer_crctc_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/resolve/main/data/lang_bpe_500/bpe.model 
```

For MoSE (Mixture-of-Speech-Encoders) model, download 4 pre-trained Zipformer model:

```bash
mkdir -p models/zipformer/librispeech/transducer_crctc_large \
        models/zipformer/librispeech/aed_crctc_large \
        models/zipformer/librispeech/crctc_large \
        models/zipformer/librispeech/transducer_large

wget -P models/zipformer/librispeech/transducer_crctc_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech/aed_crctc_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-cr-ctc-aed-20241020/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech/crctc_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-cr-ctc-20241018/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech/transducer_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/resolve/main/data/lang_bpe_500/bpe.model

```

### 1.4. Training

Baseline model:

```bash
python3 ./zipformer_llm_en/train.py \
  --world-size 4 \
  --max-duration 200 \
  --num-epochs 10 \
  --exp-dir ./zipformer_llm_en/exp \
  --manifest-dir data/fbank
```

MoSE model:

```bash
python3 ./zipformer_llm_en/train_mose.py \
  --world-size 4 \
  --max-duration 200 \
  --num-epochs 10 \
  --exp-dir ./zipformer_llm_en/exp_mose \
  --manifest-dir data/fbank
```

### 1.5. Decoding

Baseline model:

```bash
./zipformer_llm_en/decode.py \
  --exp-dir ./zipformer_llm_en/exp \
  --epoch 10 \
  --avg 6 \
  --max-duration 200 \
  --num-workers 4 \
  --manifest-dir data/fbank
``` 

MoSE model:

```bash
python3 ./zipformer_llm_en/decode_mose.py \
  --max-duration 2000 \
  --exp-dir ./zipformer_llm_en/exp_mose \
  --epoch 10 --avg 5 \
  --manifest-dir data/fbank
  ```

## 2. whisper_llm_zh

Illustrate with `icefall_asr_aishell_whisper_qwen2_1.5B` model that is trained on Aishell1 dataset.

### 2.1. Install dependencies

```bash
pip install -r whisper_llm_zh/requirements.txt
pip install huggingface_hub['cli']
```

### 2.2. Prepare data

```bash
bash whisper_llm_zh/prepare.sh
```

### 2.3. Prepare pre-trained speech encoder and LLM model

```bash

mkdir -p models/whisper models/qwen

# Download aishell fine-tuned whisper model
huggingface-cli download --local-dir models/whisper    yuekai/icefall_asr_aishell_whisper exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt

# Download Qwen2-1.5B-Instruct
huggingface-cli download  --local-dir models/qwen     Qwen/Qwen2-1.5B-Instruct
```

### 2.4. Training

```bash
# First, we only train the projector and freeze other modules.
torchrun --nproc_per_node 8 ./whisper_llm_zh/train.py \
  --max-duration 200 \
  --exp-dir ./whisper_llm_zh/exp_test \
  --speech-encoder-path-or-name models/whisper/exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt \
  --llm-path-or-name Qwen/Qwen2-1.5B-Instruct \
  --manifest-dir data/fbank \
  --deepspeed \
  --deepspeed_config ./whisper_llm_zh/ds_config_zero1.json \
  --use-flash-attn True \
  --use-lora False --unfreeze-llm False

# Then we jointly train the projector and LLM LoRA modules.
torchrun --nproc_per_node 8 ./whisper_llm_zh/train.py \
  --max-duration 200 \
  --exp-dir ./whisper_llm_zh/exp_test \
  --speech-encoder-path-or-name models/whisper/exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt \
  --llm-path-or-name Qwen/Qwen2-1.5B-Instruct \
  --manifest-dir data/fbank \
  --deepspeed \
  --deepspeed_config ./whisper_llm_zh/ds_config_zero1.json \
  --use-flash-attn True \
  --use-lora True --unfreeze-llm True
  --pretrained-model-path ./whisper_llm_zh/exp_test/epoch-3.pt
```

### 2.5. Decoding with pre-traind models

```bash
mkdir -p models/whisper models/qwen models/checkpoint

# Download pre-trained LLM-based ASR model
huggingface-cli download --local-dir models/checkpoint yuekai/icefall_asr_aishell_whisper_qwen2_1.5B

# Download pre-trained whisper model
huggingface-cli download --local-dir models/whisper    yuekai/icefall_asr_aishell_whisper exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt

# Download pre-trained Qwen2-1.5B-Instruct model
huggingface-cli download  --local-dir models/qwen     Qwen/Qwen2-1.5B-Instruct

# Create a soft link to the pre-trained LLM-based ASR model
mkdir -p whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B
ln -s models/checkpoint/epoch-10-avg-5.pt whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B/epoch-999.pt

# Decode with pre-trained LLM-based ASR model
python3 ./whisper_llm_zh/decode.py \
  --max-duration 80 \
  --exp-dir whisper_llm_zh/exp_aishell_whisper_qwen2_1.5B \
  --speech-encoder-path-or-name models/whisper/exp_large_v2/whisper-large-v2-aishell1-epoch-10-avg-6.pt  \
  --llm-path-or-name models/qwen \
  --epoch 999 --avg 1 \
  --manifest-dir data/fbank \
  --use-flash-attn True \
  --use-lora True --dataset aishell
```
