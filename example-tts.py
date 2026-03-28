#!/usr/bin/env python3

import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    model = ChatterboxTTS.from_pretrained(
        # **Update max_batch_size to account for the amount of vram you have. (keep lower value (such as 10) if you have less vram like 8gb). Increase to improve speed for environments with more vram.
        max_batch_size = 3,
        max_model_len = 1000,
    )

    for i, audio_prompt_path in enumerate([None, "docs/audio-sample-01.mp3", "docs/audio-sample-03.mp3"]):
        prompts = [
            "You are listening to a demo of the Chatterbox Turbo TTS model running on VLLM.",
            "This is a separate prompt to test the batching implementation.",
            "And here is a third prompt. It's a bit longer than the first one, but not by much.",
        ]

        audios = model.generate(
            prompts,
            audio_prompt_path=audio_prompt_path,
            diffusion_steps=2,
            top_p=0.95,
            top_k=1000,
            repetition_penalty=1.2,
        )
        for audio_idx, audio in enumerate(audios):
            ta.save(f"test-{i}-{audio_idx}.mp3", audio, model.sr)

    model.shutdown()