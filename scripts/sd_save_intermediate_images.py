import os

from modules import scripts
from modules.processing import Processed, process_images, fix_seed, create_infotext
from modules.sd_samplers import KDiffusionSampler, sample_to_image
from modules.images import save_image, FilenameGenerator, get_next_sequence_number
from modules.shared import opts

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

    def save_image_only_get_name(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
        #for description see modules.images.save_image
        
        namegen = FilenameGenerator(p, seed, prompt, image)

        if save_to_dirs is None:
            save_to_dirs = (grid and opts.grid_save_to_dirs) or (not grid and opts.save_to_dirs and not no_prompt)

        if save_to_dirs:
            dirname = namegen.apply(opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
            path = os.path.join(path, dirname)

        os.makedirs(path, exist_ok=True)

        if forced_filename is None:
            if short_filename or seed is None:
                file_decoration = ""
            elif opts.save_to_dirs:
                file_decoration = opts.samples_filename_pattern or "[seed]"
            else:
                file_decoration = opts.samples_filename_pattern or "[seed]-[prompt_spaces]"

            add_number = opts.save_images_add_number or file_decoration == ''

            if file_decoration != "" and add_number:
                file_decoration = "-" + file_decoration

            file_decoration = namegen.apply(file_decoration) + suffix

            if add_number:
                basecount = get_next_sequence_number(path, basename)
                fullfn = None
                for i in range(500):
                    fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
                    fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
                    if not os.path.exists(fullfn):
                        break
            else:
                fullfn = os.path.join(path, f"{file_decoration}.{extension}")
        else:
            fullfn = os.path.join(path, f"{forced_filename}.{extension}")

        return (fullfn)

    def process(self, p, is_active, intermediate_type, every_n):
        if is_active:
            def callback_state(self, d):
                """
                callback_state runs after each processing step
                """
                current_step = d["i"]

                if current_step % every_n == 0:
                    for index in range(0, p.batch_size):
                        if intermediate_type == "Denoised":
                            image = sample_to_image(d["denoised"], index=index)
                        else:
                            image = sample_to_image(d["x"], index=index)

                        if current_step == 0:
                            if opts.save_images_add_number:
                                digits = 5
                            else:
                                digits = 6
                            if index == 0:
                                # Set custom folder for saving intermediates on first step of first image
                                intermed_path = os.path.join(p.outpath_samples, "intermediates")
                                os.makedirs(intermed_path, exist_ok=True)
                                # Set filename with pattern. Two versions depending on opts.save_images_add_number
                                fullfn = Script.save_image_only_get_name(image, p.outpath_samples, "", int(p.seed), p.prompt, p=p)
                                base_name, _ = os.path.splitext(fullfn)
                                base_name = os.path.basename(base_name)
                                substrings = base_name.split('-')
                                if opts.save_images_add_number:
                                    intermed_number = substrings[0]
                                    intermed_number = f"{intermed_number:0{digits}}"
                                    intermed_suffix = '-'.join(substrings[1:])
                                else:
                                    intermed_number = get_next_sequence_number(intermed_path, "")
                                    intermed_number = f"{intermed_number:0{digits}}"
                                    intermed_suffix = '-'.join(substrings[0:])
                                intermed_path = os.path.join(intermed_path, intermed_number)
                                p.outpath_intermed = intermed_path
                                p.outpath_intermed_number = []
                                p.outpath_intermed_number.append(intermed_number)
                                p.outpath_intermed_suffix = intermed_suffix
                            else:
                                intermed_number = int(p.outpath_intermed_number[0]) + index
                                intermed_number = f"{intermed_number:0{digits}}"
                                p.outpath_intermed_number.append(intermed_number)

                        intermed_suffix = p.outpath_intermed_suffix.replace(str(int(p.seed)), str(int(p.all_seeds[index])), 1)
                        
                        p.outpath_intermed_pattern = p.outpath_intermed_number[index] + "-%%%-" + intermed_suffix
                        filename = p.outpath_intermed_pattern.replace("%%%", f"{current_step:03}")

                        #don't save first step
                        if current_step > 0:
                            #generate png-info
                            infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments=[], position_in_batch=index % p.batch_size, iteration=index // p.batch_size)
                            infotext = f'{infotext}, intermediate: {current_step:03d}'

                            #save intermediate image
                            save_image(image, p.outpath_intermed, "", info=infotext, p=p, forced_filename=filename)

                return orig_callback_state(self, d)

            setattr(KDiffusionSampler, "callback_state", callback_state)

    def postprocess(self, p, processed, is_active, intermediate_type, every_n):
        setattr(KDiffusionSampler, "callback_state", orig_callback_state)
