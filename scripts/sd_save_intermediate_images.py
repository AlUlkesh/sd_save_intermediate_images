import os

from modules import scripts
from modules.processing import Processed, process_images, fix_seed
from modules.sd_samplers import KDiffusionSampler, sample_to_image
from modules.images import save_image

import gradio as gr

orig_callback_state = KDiffusionSampler.callback_state


class Script(scripts.Script):
    def title(self):
        return "Save intermediate images during the sampling process"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Save intermediate images", open=False):
                with gr.Group():
                    is_active = gr.Checkbox(
                        label="Save intermediate images",
                        value=False
                    )
                with gr.Group():
                    intermediate_type = gr.Radio(
                        label="Should the intermediate images be denoised or noisy?",
                        choices=["Denoised", "Noisy"],
                        value="Denoised"
                    )
                with gr.Group():
                    every_n = gr.Number(
                        label="Save every N images",
                        value="5"
                    )
        return [is_active, intermediate_type, every_n]

    def get_next_sequence_number(path):
        from pathlib import Path
        """
        Determines and returns the next sequence number to use when saving an image in the specified directory.
        The sequence starts at 0.
        """
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                if num > result: result = num
            except ValueError:
                pass
        return result + 1

    def run(self, p, is_active, intermediate_type, every_n):
        fix_seed(p)
        return Processed(p, images, p.seed)

    def process(self, p, is_active, intermediate_type, every_n):
        if is_active:
            def callback_state(self, d):
                """
                callback_state runs after each processing step
                """
                current_step = d["i"]

                if current_step == 0:
                    # Set custom folder for saving intermediates on first step
                    intermed_path = os.path.join(p.outpath_samples, "intermediates")
                    os.makedirs(intermed_path, exist_ok=True)
                    intermed_number = Script.get_next_sequence_number(intermed_path)
                    intermed_path = os.path.join(intermed_path, f"{intermed_number:05}")
                    p.outpath_intermed = intermed_path

                if current_step % every_n == 0:
                    if intermediate_type == "Denoised":
                        image = sample_to_image(d["denoised"])
                    else:
                        image = sample_to_image(d["x"])

                    save_image(image, p.outpath_intermed, f"{current_step:03}", seed=int(p.seed), prompt=p.prompt, p=p)

                return orig_callback_state(self, d)

            setattr(KDiffusionSampler, "callback_state", callback_state)

    def postprocess(self, p, processed, is_active, intermediate_type, every_n):
        setattr(KDiffusionSampler, "callback_state", orig_callback_state)
