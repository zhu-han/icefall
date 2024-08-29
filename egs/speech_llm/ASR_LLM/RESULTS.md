## Results

### zipformer_llm_en results

|Model|         Training Dataset  | Speech Encoder | LLM |  Projector |
|-| -------------------------| ----------------|------|---------------|
|[Zipformer + LLM](https://huggingface.co/zhu-han/icefall_asr_librispeech_zipformer_qwen2_1.5B)  | LibriSpeech  | Zipformer Large Transducer/CR-CTC, freeze| Qwen2-1.5B-Instruct, LoRA | Linear, 4x downsample|
|[Zipformer MoSE + LLM](https://huggingface.co/zhu-han/icefall_asr_librispeech_zipformer_mose_qwen2_1.5B)  | LibriSpeech  | Mixture-of-Speech-Encoders (4 Zipformer Large models), freeze| Qwen2-1.5B-Instruct, LoRA | Linear, 4x downsample|


WER Details:

| Model | LibriSpeech test-clean | LibriSpeech test-other | 
|-------|---------------------|-----------|
|[Zipformer + LLM](https://huggingface.co/zhu-han/icefall_asr_librispeech_zipformer_qwen2_1.5B) | 1.84 | 3.83 |
|[Zipformer MoSE + LLM](https://huggingface.co/zhu-han/icefall_asr_librispeech_zipformer_mose_qwen2_1.5B) | 1.74 | 3.53 |


### whisper_llm_zh results

|Model|         Training Dataset  | Speech Encoder | LLM |  Projector |
|-| -------------------------| ----------------|------|---------------|
|[yuekai/icefall_asr_aishell_whisper_qwen2_1.5B](https://huggingface.co/yuekai/icefall_asr_aishell_whisper_qwen2_1.5B)  | Aishell1                | whisper-large-v2-aishell1-ft, freeze| Qwen2-1.5B-Instruct, LoRA | Linear, 8x downsample|
| [yuekai/icefall_asr_multi-hans_whisper_qwen2_1.5B](https://huggingface.co/yuekai/icefall_asr_multi-hans_whisper_qwen2_1.5B)  |Multi-hans-zh                | whisper-large-v2-multi-hans-ft, freeze| Qwen2-1.5B-Instruct, LoRA | Linear, 8x downsample||
| [yuekai/icefall_asr_multi-hans_whisper_qwen2_7B](https://huggingface.co/yuekai/icefall_asr_multi-hans_whisper_qwen2_7B)  |Multi-hans-zh                | whisper-large-v2-multi-hans-ft, freeze| Qwen2-7B-Instruct, LoRA | Linear, 8x downsample||

CER Details:
| Model | [yuekai/icefall_asr_aishell_whisper_qwen2_1.5B](https://huggingface.co/yuekai/icefall_asr_aishell_whisper_qwen2_1.5B) | [yuekai/icefall_asr_multi-hans_whisper_qwen2_1.5B](https://huggingface.co/yuekai/icefall_asr_multi-hans_whisper_qwen2_1.5B) | [yuekai/icefall_asr_multi-hans_whisper_qwen2_7B](https://huggingface.co/yuekai/icefall_asr_multi-hans_whisper_qwen2_7B) |
|-------|------------------------------------------------|----------------------------------------------------|-|
| Split | Greedy Search | Greedy Search | Greedy Search |
| aishell-1 dev | - | 0.66 | 0.49|
| aishell-1 test | 3.62 | 0.68 | 0.51 |
| aishell-2 dev | - | 2.67 | 2.61 |
| aishell-2 test | - | 2.94 | 2.76 |
| aishell-4 test | - | 16.20 | 15.82 |
| alimeeting eval | - | 30.86 | 29.27 |
| alimeeting test | - | 40.50 | 39.48 |
| magicdata dev | - | 2.50 | 2.27 |
| magicdata test | - | 1.70 | 1.57 |
| kespeech-asr dev phase1 | - | 6.22 | 4.87 |
| kespeech-asr dev phase2 | - | 2.18 | 1.87 |
| kespeech-asr test | - | 6.59 | 5.76 |
| WenetSpeech dev | - | 4.59 | 4.41 |
| WenetSpeech test_meeting | - | 6.41 | 6.06 |
| WenetSpeech tes_net | - | 6.63 | 6.30 |
| SPEECHIO Avg 001-026 | - | 4.80 | 4.50 |