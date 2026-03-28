import random
import os

import numpy as np
import torch
import gradio as gr
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS

DEVICE = "cuda"

config_seed = None
global_model = None

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    global config_seed
    config_seed = seed


def load_model():
    print("Loading model...")
    global global_model
    global_model = ChatterboxTTS.from_pretrained(
        # **Update max_batch_size to account for the amount of vram you have. (keep lower value (such as 10) if you have less vram like 8gb). Increase to improve speed for environments with more vram.
        max_batch_size = 15,
        max_model_len = 1000,
    )
    return global_model

def generate(text, audio_prompt_path, temperature, seed_num,
             diffusion_steps,
             top_k, top_p, repetition_penalty):
    if seed_num != 0:
        set_seed(int(seed_num))

    print(f"Using text: {text}")
    print(f"Using audio_prompt_path: {audio_prompt_path}")
    print(f"Using seed: {config_seed}")
    print(f"Using temperature: {temperature}")
    print(f"Using top_k: {top_k}")
    print(f"Using top_p: {top_p}")
    print(f"Using repetition_penalty: {repetition_penalty}")

    wav = global_model.generate(
        [text],
        audio_prompt_path=audio_prompt_path,
        temperature=temperature,
        diffusion_steps=diffusion_steps,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=config_seed,
    )
    return (global_model.sr, wav[0].squeeze(0).numpy())


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                diffusion_steps = gr.Slider(1, 10, step=1, label="Diffusion Steps (turbo default: 2)", value=2)
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                top_k = gr.Slider(0, 5000, step=10, label="top_k", value=1000)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=0.95)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            ref_wav,
            temp,
            seed_num,
            diffusion_steps,
            top_k,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    # Don't let Gradio manage the model loading, it's causing issues.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    load_model()

    print("Starting Gradio app...")
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
