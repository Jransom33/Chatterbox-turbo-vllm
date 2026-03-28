#!/usr/bin/env python3

import torch
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS

AUDIO_PROMPT_PATH = "docs/audio-sample-01.mp3"
MAX_MODEL_LEN = 1000 # Maximum length of generated audio in tokens

if __name__ == "__main__":
    # Print current GPU memory usage
    print(f"[START] Starting GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    model = ChatterboxTTS.from_pretrained(
        # Only allocate enough memory for a single request.
        # **Update max_batch_size to account for the amount of vram you have. (keep lower value (such as 10) if you have less vram like 8gb). Increase to improve speed for environments with more vram.
        max_batch_size = 10,
        max_model_len = MAX_MODEL_LEN,
    )

    print(f"[POST-INIT] GPU memory usage after model load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Generate audio conditioning
    # The resulting s3gen_ref and cond_emb can be reused for multiple generations, or saved/loaded from disk.
    s3gen_ref, cond_emb = model.get_audio_conditionals(AUDIO_PROMPT_PATH)

    # Generate audio
    audios = model.generate_with_conds(
        ["You are listening to a demo of the Chatterbox TTS model running on VLLM."],
        s3gen_ref=s3gen_ref,
        cond_emb=cond_emb,
        diffusion_steps=2,
        top_p=0.95,
        top_k=1000,
        repetition_penalty=1.2,
    )
    print(f"[POST-GEN] GPU memory usage after generating audio: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    for audio_idx, audio in enumerate(audios):
        ta.save(f"test-{audio_idx}.mp3", audio, model.sr)

    model.shutdown()