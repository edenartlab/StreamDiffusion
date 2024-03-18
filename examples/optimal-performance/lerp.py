import os
import sys
import time
from multiprocessing import Process, Queue, get_context
from typing import Literal
import numpy as np
import fire
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper


def compute_embedding_trajectory(stream,
    n_frames_between_prompts = 10):
    
    prompts = [
        "breathtakingly beautiful ultrawide angle colour masterpiece of a creepy family by roger dean and greg hildebrandt and kilian eng and jean giraud and beeple, forest clearing, lake, reflection, symmetry, secret overgrown temple, mysterious person looking over shoulder, incredible sense of depth and perspective and clarity, arch, weird abstract, 8 k",
        'dreamlike photo of a majestic bed floating above the floor in a giant red marble room with windows opening to eternity, clouds, by andrzej sykut by lee madgewick, photorealistic, octane render, recursive flowing, cascading, multiverse labyrinthine, optical illusion, impossible angles',
        "elegant minimalist metal wire art of symmetrical and expressive female facial features and silhouette ",
        "friendly dryad tree creature, in a forest, golden hour, professional photography, mild depth of field, cinematic lighting, 8 k ",
        "magic stone portal in the forest, cinematic lighting, dramatic, octane render, long lens, shallow depth of field, bokeh, anamorphic lens flare, 8k, hyper detailed",
        'scary fairy tale mystical forest twisted creepy trees magical forest landscape artwork trees near the path amazing nature, surreal, dreamlike, lucid dream, very detailed, perfect lighting, perfect composition, 4 k, artgerm, derek zabrocki, greg rutkowski',
        "the black lioness made of tar, bathing in the bathtub filled with tar, dripping tar, drooling goo, sticky black goo, photography, dslr, reflections, black goo, rim lighting, modern bathroom, hyper realistic, 8k resolution, unreal engine 5, raytracing",
        "a great, twisted tree growing in an abandoned brutalist bunker, in the round, skylight with light shining down, snow falling through skylight, photorealistic",
        "a huge woman head trapped in stone in an epic landscape, cinematic light, unreal engine, trending on artstation",
        "a twisted trending on art shards of sharp 5, raytracing photorealistic, digital painting, irrational fear the existential dread f 3 black glass lenses floats through futuristic illustrated visionary",
        "barlow, bao volumetric cinematic lighting, 8 k smooth puffy sculptural big leaves and pham, donato giancola, painting, aesthetic, smooth, bed above floor",
        "a closeup portrait of a woman wrapped in a translucent sphere, standing next to a giant huge levitating copper orb, in a foggy pond, golden hour, color photograph, by vincent desiderio, canon eos c 3 0 0, ƒ 1. 8, 3 5 mm, 8 k",
        "style, volumetric and gold, cascading, multiverse a beautiful greg rutkowski, oil paint, art, lifelike, photorealistic, reclections, reflective, sharp details, 4k, trending on artstation",
        "a woman with her face painted like the wings of a butterfly. butterfly wings face painting. beautiful highly detailed face. painting by artgerm and greg rutkowski and alphonse mucha.",
        "fine artwork of a still life arrangement of flowering nasturtiums and time crystals, soft light coming from a window in a dark room, moody, beautiful classical style",
        "incredible photograph of retro interier of glass museum, full of hundreds of massive glass television screens from different places of the universe, incredible sharp details, back light contrast, dramatic dark atmosphere, bright vivid colours",
        "surreal baroque portrait of ribbed white alien god covered with spinal antlers, on white exoplanet, iredescent soft glow bloom effect, dream - like, baroque portrait painting, perfect composition, beautiful detailed intricate detailed octane render",
    ]

    print(f"Encoding {len(prompts)} prompts...")
    prompt_embeds = [stream.stream.get_prompt_embeds(prompt) for prompt in prompts]
    prompt_embeds.append(prompt_embeds[0]) # Add the first prompt again to close the loop
    
    final_frame_prompt_embeds = []
    for prompt_i in range(len(prompts)):
        for step in range(n_frames_between_prompts):
            interpolation_f = step / n_frames_between_prompts
            frame_embed = prompt_embeds[prompt_i] * (1 - interpolation_f) + prompt_embeds[prompt_i + 1] * interpolation_f
            final_frame_prompt_embeds.append(frame_embed)

    print(f"Frame embeds shape: {len(final_frame_prompt_embeds)} x {final_frame_prompt_embeds[0].shape}")

    return final_frame_prompt_embeds


class PromptUpdater:
    def __init__(self, 
        stream: StreamDiffusionWrapper,
        n_frames_between_prompts: int = 48,
        ):
        self.stream = stream
        self.embedding_index = 0
        self.n_frames_between_prompts = n_frames_between_prompts

        self.previous_prompt = "darkness, the void, eerie silence, liminal space"
        self.current_prompt  = "a mystical oracle, floating in the void, veiled woman with deep blue eyes"

        self.previous_prompt_encoding = stream.stream.get_prompt_embeds(self.previous_prompt)
        self.current_prompt_encoding  = stream.stream.get_prompt_embeds(self.current_prompt)

        self.interpolate_trajectory(self.previous_prompt_encoding, self.current_prompt_encoding)
    
    def sync_prompt(self):
        prompt_txt_path = "/data/xander/Projects/cog/GitHub_repos/StreamDiffusion/speech2speech/prompt.txt"
        with open(prompt_txt_path, "r") as file:
            new_prompt = file.read()

        prompt_changed = new_prompt != self.current_prompt

        if prompt_changed:
            self.previous_prompt = self.current_prompt
            self.current_prompt = new_prompt

        return prompt_changed

    def interpolate_trajectory(self, embed1, embed2):
        # interpolate between the previous and current prompt encodings:
        self.new_prompt_embeds = []
        for step in range(self.n_frames_between_prompts):
            interpolation_f = step / self.n_frames_between_prompts
            frame_embed = embed1 * (1 - interpolation_f) + embed2 * interpolation_f
            self.new_prompt_embeds.append(frame_embed)

    def update_conditions(self, frame_id, frame_embeds):
        prompt_changed = self.sync_prompt()

        if prompt_changed:
            self.current_prompt_encoding = self.stream.stream.get_prompt_embeds(self.current_prompt)
            current_frame_condition = self.new_prompt_embeds[min(self.embedding_index, len(self.new_prompt_embeds) - 1)]
            self.interpolate_trajectory(current_frame_condition, self.current_prompt_encoding)
            self.embedding_index = 0

            print("----------------------------------------------------------------")
            print("New target: \n", self.current_prompt)
            print("----------------------------------------------------------------")

        self.embedding_index += 1

        # Update the frame condition:
        try:
            current_frame_condition = self.new_prompt_embeds[min(self.embedding_index, len(self.new_prompt_embeds) - 1)]
            self.stream.stream.update_prompt_embeds(current_frame_condition)
        except:
            pass


def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    prompt: str,
    model_id_or_path: str,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    prompt : str
        The prompt to generate images from.
    model_id_or_path : str
        The name of the model to use for image generation.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    """

    if "turbo" in model_id_or_path:
        stream = StreamDiffusionWrapper(
            model_id_or_path=model_id_or_path,
            lora_dict=None,
            t_index_list=[0],
            frame_buffer_size=1,
            width=width,
            height=height,
            warmup=10,
            acceleration=acceleration,
            use_lcm_lora=False,
            mode="txt2img",
            use_denoising_batch=True,
            cfg_type="none",
            seed = 0,
    )
    else:
        stream = StreamDiffusionWrapper(
            model_id_or_path=model_id_or_path,
            lora_dict=None,
            t_index_list=[0, 16, 32, 45],
            frame_buffer_size=1,
            width=width,
            height=height,
            warmup=10,
            acceleration=acceleration,
            use_lcm_lora=True,
            mode="txt2img",
            use_denoising_batch=True,
            cfg_type="none",
            seed = 0,
    )


    prompt_updater = PromptUpdater(stream)

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    frame_embeds = compute_embedding_trajectory(stream)
    frame_id, loop_index = 0, 0
    print("LFG...")

    while True:
        try:
            start_time = time.time()

            noise_seed = loop_index + int(frame_id / len(frame_embeds))
            prompt_updater.update_conditions(frame_id, frame_embeds)

            period = 12 # (frames)
            noise_inject_amplitude = 0.0 * np.sin(2 * np.pi * frame_id / period)
            #print(f"noise_inject_amplitude: {noise_inject_amplitude}")
            #x_outputs = stream.stream.txt2img_sd_turbo(1).cpu()
            x_outputs = stream.stream.lerp_step(1, noise_seed, noise_inject_amplitude).cpu()

            queue.put(x_outputs, block=False)

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
            frame_id += 1

        except KeyboardInterrupt:
            print(f"fps: {fps}")
            return

def main(
    prompt: str = "magical portal into the underworld, lush forest, photoreal, 8K",
    #model_id_or_path: str = "stabilityai/sd-turbo",
    model_id_or_path: str = "Lykon/dreamshaper-8",
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    width: int = 768,
    height: int = 512,
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """

    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    process1 = ctx.Process(
        target=image_generation_process,
        args=(queue, fps_queue, prompt, model_id_or_path, width, height, acceleration),
    )
    process1.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue, width, height))
    process2.start()

    process1.join()
    process2.join()

if __name__ == "__main__":
    fire.Fire(main)
