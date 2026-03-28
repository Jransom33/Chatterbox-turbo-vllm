from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import time

from vllm import LLM, SamplingParams
from functools import lru_cache

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.s3gen.const import S3GEN_SIL
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .text_utils import punc_norm

REPO_ID = "ResembleAI/chatterbox-turbo"

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 15 * S3_SR  # 15 seconds of reference audio for conditioning (turbo uses longer)
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3: LLM, t3_config: T3Config, t3_cond_enc: T3CondEnc,
                 t3_speech_emb: torch.nn.Embedding,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals):
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.t3 = t3
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb

        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda",
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,

                   # Quantization options: None, "bitsandbytes", "fp8" (requires Ampere+ GPU)
                   quantization: Optional[str] = None,

                   s3gen_use_fp16: bool = False,
                   **kwargs) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        t3_config = T3Config()

        # Load weights for T3CondEnc and speech_emb (needed for conditioning outside vLLM)
        t3_weights = load_file(ckpt_dir / "t3_turbo_v1.safetensors")

        t3_enc = T3CondEnc(t3_config)
        t3_enc.load_state_dict({ k.replace('cond_enc.', ''):v for k,v in t3_weights.items() if k.startswith('cond_enc.') })
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        t3_speech_emb.load_state_dict({ k.replace('speech_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_emb.') })
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()

        # Heuristic: rough calculation for what percentage of GPU memory to give to vLLM.
        # Turbo model is ~350M params, slightly smaller than the original 500M.
        vllm_memory_needed = (1.2*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        # Symlink weights into the config directory for vLLM
        t3_turbo_path = ckpt_dir / "t3_turbo_v1.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model-turbo" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_turbo_path)

        # Copy tokenizer files into config directory for vLLM
        import shutil
        turbo_config_dir = Path.cwd() / "t3-model-turbo"
        for tok_file in ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]:
            src = ckpt_dir / tok_file
            if src.exists():
                dst = turbo_config_dir / tok_file
                if not dst.exists() or not dst.is_symlink():
                    dst.unlink(missing_ok=True)
                    shutil.copy2(src, dst)

        base_vllm_kwargs = {
            "model": "./t3-model-turbo",
            "task": "generate",
            "tokenizer": "./t3-model-turbo",
            "tokenizer_mode": "auto",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
            # Disable chunked prefill: the custom T3 architecture uses positional
            # multimodal embeddings that break when vllm splits prefill mid-sequence.
            "enable_chunked_prefill": False,
        }

        # Add quantization if specified
        if quantization:
            base_vllm_kwargs["quantization"] = quantization
            if quantization == "bitsandbytes":
                base_vllm_kwargs["load_format"] = "bitsandbytes"
            print(f"Using {quantization} quantization for T3 model")

        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16, meanflow=True)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        default_conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            default_conds = Conditionals.load(builtin_voice)
            default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
        )

    @classmethod
    def from_pretrained(cls, quantization: Optional[str] = None, *args, **kwargs) -> 'ChatterboxTTS':
        import os
        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or None,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, quantization=quantization, *args, **kwargs)

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        if wav_fpath is None:
            assert self.default_conds is not None, "No default conditionals loaded. Please provide audio_prompt_path."
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            ## Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens - pad audio to ENC_COND_LEN to guarantee exactly speech_cond_prompt_len tokens
            import numpy as np
            enc_cond_wav = ref_16k_wav[:self.ENC_COND_LEN]
            if len(enc_cond_wav) < self.ENC_COND_LEN:
                enc_cond_wav = np.pad(enc_cond_wav, (0, self.ENC_COND_LEN - len(enc_cond_wav)))
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([enc_cond_wav], max_len=self.t3_config.speech_cond_prompt_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        # For turbo: no perceiver resampler, just raw speech embeddings
        cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0]

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=torch.ones(0),  # No emotion conditioning in turbo
        ).to(device=self.target_device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def generate(
        self,
        prompts: Union[str, list[str]],
        audio_prompt_path: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens=1000, # Capped at max_model_len

        top_p=0.95,
        top_k=1000,
        repetition_penalty=1.2,

        # Number of diffusion steps to use for S3Gen
        # Turbo defaults to 2 meanflow steps.
        diffusion_steps: int = 2,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return self.generate_with_conds(
            prompts=prompts,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            diffusion_steps=diffusion_steps,
            *args, **kwargs
        )

    def generate_with_conds(
        self,
        prompts: Union[str, list[str]],
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        temperature: float = 0.8,
        max_tokens=1000, # Capped at max_model_len

        # Number of diffusion steps (turbo default: 2)
        diffusion_steps: int = 2,

        top_p=0.95,
        top_k=1000,
        repetition_penalty=1.2,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        if isinstance(prompts, str):
            prompts = [prompts]

        # Norm text (turbo uses GPT2 tokenizer, no [START]/[STOP] wrapping needed)
        prompts = [punc_norm(p) for p in prompts]

        with torch.inference_mode():
            start_time = time.time()
            batch_results = self.t3.generate(
                [
                    {
                        "prompt": text,
                        "multi_modal_data": {
                            "conditionals": [cond_emb],
                        },
                    }
                    for text in prompts
                ],
                sampling_params=SamplingParams(
                    temperature=temperature,

                    stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,

                    *args, **kwargs,
                )
            )
            t3_gen_time = time.time() - start_time
            print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

            # run torch gc
            torch.cuda.empty_cache()

            start_time = time.time()
            results = []
            for i, batch_result in enumerate(batch_results):
                for output in batch_result.outputs:
                    if i % 5 == 0:
                        print(f"[S3] Processing prompt {i} of {len(batch_results)}")

                    # Run gc every 10 prompts
                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                    speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in output.token_ids], device="cuda")
                    # Remove OOV tokens
                    speech_tokens = speech_tokens[speech_tokens < 6561]
                    # Add silence padding at end
                    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(speech_tokens.device)
                    speech_tokens = torch.cat([speech_tokens, silence])

                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=s3gen_ref,
                        n_cfm_timesteps=diffusion_steps,
                    )
                    results.append(wav.cpu())
            s3gen_gen_time = time.time() - start_time
            print(f"[S3Gen] Waveform Generation time: {s3gen_gen_time:.2f}s")

            return results

    def shutdown(self):
        del self.t3
        torch.cuda.empty_cache()
