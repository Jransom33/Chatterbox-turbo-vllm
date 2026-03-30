# Chatterbox Turbo TTS on vLLM

> This project is a port of [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) to vLLM, built on top of the excellent [chatterbox-vllm](https://github.com/randombk/chatterbox-vllm) by [@randombk](https://github.com/randombk). The original chatterbox-vllm project ported the standard Chatterbox model to vLLM — this repo extends that work to support the newer **Chatterbox Turbo** model, which features a significantly faster S3Gen waveform decoder.

- Improved performance and more efficient use of GPU memory.
- Easier integration with state-of-the-art inference infrastructure.

DISCLAIMER: THIS IS A PERSONAL PROJECT and is not affiliated with my employer or any other corporate entity in any way. The project is based solely on publicly-available information. All opinions are my own and do not necessarily represent the views of my employer.

## Chatterbox VLLM vs Turbo VLLM (RTX 4090, BATCH_SIZE=80)


| Metric                       | Regular             | Turbo               | Speedup  |
| ---------------------------- | ------------------- | ------------------- | -------- |
| Audio duration               | 39.9 min            | 38.5 min            | —        |
| Model load                   | 27.3s               | 21.4s               | 1.3x     |
| **Generation**               | **103.1s**          | **61.3s**           | **1.7x** |
| — T3 speech token generation | 31.6s               | 39.9s               | 0.8x     |
| — S3Gen waveform generation  | 70.4s               | 20.2s               | 3.5x     |
| End-to-end total             | 131.1s              | 83.3s               | 1.6x     |
| **Generation RTF**           | **23.2x real-time** | **37.6x real-time** | **1.6x** |
| **End-to-end RTF**           | **18.3x real-time** | **27.7x real-time** | **1.5x** |


# Project Status: Usable and with Benchmark-Topping Throughput

- ✅ Basic speech cloning with audio and text conditioning.
- ✅ Outputs match the quality of the original Chatterbox implementation.
- ✅ Context Free Guidance (CFG) is implemented.
  - Due to a vLLM limitation, CFG can not be tuned on a per-request basis and can only be configured via the `CHATTERBOX_CFG_SCALE` environment variable.
- ✅ Exaggeration control is implemented.
- ✅ vLLM batching is implemented and produces a significant speedup.
- ℹ️ Project uses vLLM internal APIs and extremely hacky workarounds to get things done.
  - Refactoring to the idiomatic vLLM way of doing things is WIP, but will require some changes to vLLM.
  - Until then, this is a Rube Goldberg machine that will likely only work with vLLM 0.9.2.
  - Follow [https://github.com/vllm-project/vllm/issues/21989](https://github.com/vllm-project/vllm/issues/21989) for updates.
- ℹ️ Substantial refactoring is needed to further clean up unnecessary workarounds and code paths.
- ℹ️ Server API is not implemented and will likely be out-of-scope for this project.
- ❌ Learned speech positional embeddings are not applied, pending support in vLLM. However, this doesn't seem to be causing a very noticeable degradation in quality.
- ❌ APIs are not yet stable and may change.
- ❌ Benchmarks and performance optimizations are not yet implemented.

# Installation

This project only supports Linux and WSL2 with Nvidia hardware. AMD *may* work with minor tweaks, but is not tested.

Prerequisites: `git` and `[uv](https://pypi.org/project/uv/)` must be installed

```
git clone https://github.com/randombk/chatterbox-vllm.git
cd chatterbox-vllm
uv venv
source .venv/bin/activate
uv sync
```

**Blackwell GPUs (RTX 5090, etc.)** require PyTorch with CUDA 12.8 support. Use the `blackwell` extra:

```
uv sync --extra blackwell
```

The package should automatically download the correct model weights from the Hugging Face Hub.

If you encounter CUDA issues, try resetting the venv and using `uv pip install -e .` instead of `uv sync`.

# Updating

If you are updating from a previous version, run `uv sync` to update the dependencies. The package will automatically download the correct model weights from the Hugging Face Hub.

# Example

[This example](https://github.com/randombk/chatterbox-vllm/blob/master/example-tts.py) can be run with `python example-tts.py` to generate audio samples for three different prompts using three different voices.

```python
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    model = ChatterboxTTS.from_pretrained(
        gpu_memory_utilization = 0.4,
        max_model_len = 1000,

        # Disable CUDA graphs to reduce startup time for one-off generation.
        enforce_eager = True,
    )

    for i, audio_prompt_path in enumerate([None, "docs/audio-sample-01.mp3", "docs/audio-sample-03.mp3"]):
        prompts = [
            "You are listening to a demo of the Chatterbox TTS model running on VLLM.",
            "This is a separate prompt to test the batching implementation.",
            "And here is a third prompt. It's a bit longer than the first one, but not by much.",
        ]
    
        audios = model.generate(prompts, audio_prompt_path=audio_prompt_path, exaggeration=0.8)
        for audio_idx, audio in enumerate(audios):
            ta.save(f"test-{i}-{audio_idx}.mp3", audio, model.sr)
```

# Benchmarks

To run a benchmark, tweak and run `benchmark.py`.  
The following results were obtained with batching on a 6.6k-word input (`docs/benchmark-text-1.txt`), generating ~40min of audio.

## Benchmark: RTX 4090

System Specs:

- RTX 4090: 24GB VRAM (Rented on RunPod)
- 6 vCPU
- 41 GB RAM

Settings & Results:

- Input text: `docs/benchmark-text-1.txt` (6.6k words, 154 chunks)
- Input audio: `docs/audio-sample-03.mp3`
- Batch size: 80, Diffusion steps: 2, Temperature: 0.8
- CUDA graphs disabled
- Generated output length: 38m29s
- Wall time: 1m23s
- Generation time (without model startup time): 61.3s
  - Time spent in T3 speech token generation: 39.9s
  - Time spent in S3Gen waveform generation: 20.2s

Logs:

```
[BENCHMARK] Text chunked into 154 chunks
Fetching 10 files: 100%|████████████████████████████| 10/10 [00:00<00:00, 59158.03it/s]
Giving vLLM 54.97% of GPU memory (13228.80 MB)
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 03-27 22:26:07 [config.py:1604] Using max model len 1200
INFO 03-27 22:26:08 [config.py:2434] Chunked prefill is enabled with max_num_batched_tokens=8192.
WARNING 03-27 22:26:08 [__init__.py:2899] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing for more information. Reason: CUDA is initialized
INFO 03-27 22:26:13 [__init__.py:235] Automatically detected platform cuda.
INFO 03-27 22:26:17 [core.py:572] Waiting for init message from front-end.
INFO 03-27 22:26:17 [core.py:71] Initializing a V1 LLM engine (v0.10.0) with config: model='./t3-model-turbo', speculative_config=None, tokenizer='./t3-model-turbo', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=1200, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=./t3-model-turbo, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":0,"local_cache_dir":null}
INFO 03-27 22:26:17 [parallel_state.py:1102] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
WARNING 03-27 22:26:17 [topk_topp_sampler.py:59] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
INFO 03-27 22:26:17 [gpu_model_runner.py:1843] Starting to load model ./t3-model-turbo...
INFO 03-27 22:26:18 [gpu_model_runner.py:1875] Loading model from scratch...
INFO 03-27 22:26:18 [cuda.py:290] Using Flash Attention backend on V1 engine.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 112.10it/s]

INFO 03-27 22:26:21 [default_loader.py:262] Loading weights took 3.02 seconds
INFO 03-27 22:26:21 [gpu_model_runner.py:1892] Model loading took 0.7002 GiB and 3.055451 seconds
INFO 03-27 22:26:21 [gpu_model_runner.py:2380] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 21 conditionals items of the maximum feature size.
INFO 03-27 22:26:22 [gpu_worker.py:255] Available KV cache memory: 11.67 GiB
INFO 03-27 22:26:22 [kv_cache_utils.py:833] GPU KV cache size: 127,472 tokens
INFO 03-27 22:26:22 [kv_cache_utils.py:837] Maximum concurrency for 1,200 tokens per request: 106.23x
INFO 03-27 22:26:23 [core.py:193] init engine (profile, create kv cache, warmup model) took 1.29 seconds
/usr/local/lib/python3.11/dist-packages/diffusers/models/lora.py:391: FutureWarning: `LoRACompatibleLinear` is deprecated and will be removed in version 1.0.0. Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.
  deprecate("LoRACompatibleLinear", "1.0.0", deprecation_message)
[BENCHMARK] Model loaded in 21.357989072799683 seconds
Adding requests: 100%|████████████████████████████████████| 154/154 [00:01<00:00, 124.94it/s]
Processed prompts: 100%|█| 154/154 [00:38<00:00,  3.98it/s, est. speed input: 1718.66 toks/s,
[T3] Speech Token Generation time: 39.94s
[S3] Processing prompt 0 of 154
[S3] Processing prompt 5 of 154
...
[S3] Processing prompt 150 of 154
[S3Gen] Waveform Generation time: 20.23s
[BENCHMARK] Generation completed in 61.34 seconds
[BENCHMARK] Audio duration: 2308.88 seconds (38.5 min)
[BENCHMARK] Generation RTF: 37.6x real-time
[BENCHMARK] Total time: 83.26320457458496 seconds
```

Summary (BATCH_SIZE=80, RTX 4090):

## Chatterbox Turbo VLLM (RTX4090)

| Metric                       | Value                   |
| ---------------------------- | ----------------------- |
| Input text                   | 6.6k words (154 chunks) |
| Generated audio              | 38.5 min                |
| Model load                   | 21.4s                   |
| Generation time              | 61.3s                   |
| — T3 speech token generation | 39.9s                   |
| — S3Gen waveform generation  | 20.2s                   |
| **Generation RTF**           | **37.6x real-time**     |
| End-to-end total             | 83.3s                   |
| **End-to-end RTF**           | **27.7x real-time**     |


# Chatterbox Architecture

I could not find an official explanation of the Chatterbox architecture, so below is my best explanation based on the codebase. Chatterbox broadly follows the [CosyVoice](https://funaudiollm.github.io/cosyvoice2/) architecture, applying intermediate fusion multimodal conditioning to a 0.5B parameter Llama model.

![Chatterbox Architecture Diagram](docs/chatterbox-architecture.svg)

# Implementation Notes

## CFG Implementation Details

vLLM does not support CFG natively, so substantial hacks were needed to make it work. At a high level, we trick vLLM into thinking the model has double the hidden dimension size as it actually does, then splitting and restacking the states to invoke Llama with double the original batch size. This does pose a risk that vLLM will underestimate the memory requirements of the model - more research is needed into whether vLLM's initial profiling pass will capture this nuance.

![vLLM CFG Implementation](docs/vllm-cfg-impl.svg)

#
