import logging
import os
import platform
import re
import sys

from modules import paths
from modules import scripts
from modules.processing import Processed, process_images, fix_seed, create_infotext
from modules.sd_samplers import KDiffusionSampler, sample_to_image
from modules.images import save_image, FilenameGenerator, get_next_sequence_number
from modules.shared import opts, state, cmd_opts

from ffmpy import FFmpeg
import gradio as gr; gr.__version__

orig_callback_state = KDiffusionSampler.callback_state

class Script(scripts.Script):
    def title(self):
        return "Save intermediate images during the sampling process"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Save intermediate images", open=False):
            with gr.Row():
                ssii_is_active = gr.Checkbox(
                    label="Save intermediate images",
                    value=False
                )
            with gr.Row():
                ssii_intermediate_type = gr.Radio(
                    label="Should the intermediate images be denoised or noisy?",
                    choices=["Denoised", "Noisy"],
                    value="Denoised"
                )
            with gr.Row():
                ssii_every_n = gr.Number(
                    label="Save every N images",
                    value="5"
                )
            with gr.Row():
                ssii_stop_at_n = gr.Number(
                    label="Stop at N images (must be 0 = don't stop early or a multiple of 'Save every N images')",
                    value="0"
                )
            with gr.Box():
                with gr.Row():
                    ssii_video = gr.Checkbox(
                        label="Make a video file",
                        value=False
                    )
                with gr.Row():
                    ssii_video_format = gr.Radio(
                        label="Format",
                        choices=["mp4", "webm", "gif"],
                        value="mp4"
                    )
                    ssii_video_fps = gr.Number(
                        label="fps",
                        value=2
                    )
                    ssii_video_hires = gr.Radio(
                        label="If Hires. fix, use only pass",
                        choices=["1", "2"],
                        value="2"
                    )
            with gr.Row():
                ssii_debug = gr.Checkbox(
                    label="Debug",
                    value=False
                )
        with gr.Row():
            gr.HTML('<div style="padding-bottom: 0.7em;"></div><div></div>')
        return [ssii_is_active, ssii_intermediate_type, ssii_every_n, ssii_stop_at_n, ssii_video, ssii_video_format, ssii_video_fps, ssii_video_hires, ssii_debug]

    def save_image_only_get_name(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
        #for description see modules.images.save_image, same code up saving of files
        
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

    def process(self, p, ssii_is_active, ssii_intermediate_type, ssii_every_n, ssii_stop_at_n, ssii_video, ssii_video_format, ssii_video_fps, ssii_video_hires, ssii_debug):
        if ssii_is_active:

            # Debug logging
            if ssii_debug:
                mode = logging.DEBUG
                logging.basicConfig(level=mode, format='%(asctime)s %(levelname)s %(message)s')
            else:
                mode = logging.WARNING
            logger = logging.getLogger(__name__)
            logger.setLevel(mode)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{sys.executable} {sys.version}")
                logger.debug(f"{platform.system()} {platform.version()}")
                try:
                    git = os.environ.get('GIT', "git")
                    commit_hash = os.popen(f"{git} rev-parse HEAD").read()
                except Exception as e:
                    commit_hash = e
                logger.debug(f"{commit_hash}")
                logger.debug(f"Gradio {gr.__version__}")
                logger.debug(f"{paths.script_path}")
                with open(cmd_opts.ui_config_file, "r") as f:
                    logger.debug(f.read())
                with open(cmd_opts.ui_settings_file, "r") as f:
                    logger.debug(f.read())

            def callback_state(self, d):
                """
                callback_state runs after each processing step
                """
                current_step = d["i"]

                if hasattr(p, "enable_hr"):
                    hr = p.enable_hr
                else:
                    hr = False 

                logger.debug("ssii_intermediate_type, ssii_every_n, ssii_stop_at_n, ssii_video, ssii_video_format, ssii_video_fps, ssii_video_hires, ssii_debug:")
                logger.debug(f"{ssii_intermediate_type}, {ssii_every_n}, {ssii_stop_at_n}, {ssii_video}, {ssii_video_format}, {ssii_video_fps}, {ssii_video_hires}, {ssii_debug}")
                logger.debug(f"Step: {current_step}")
                logger.debug(f"hr: {hr}")

                #Highres. fix requires 2 passes
                if not hasattr(p, 'intermed_final_pass'):
                    if hr:
                        p.intermed_first_pass = True
                        p.intermed_final_pass = False
                    else:
                        p.intermed_first_pass = True
                        p.intermed_final_pass = True

                #Check if pass 1 has finished
                if hasattr(p, 'intermed_max_step'):
                    if current_step >= p.intermed_max_step:
                        p.intermed_max_step = current_step
                    else:
                        p.intermed_first_pass = False
                        p.intermed_final_pass = True
                        p.intermed_max_step = current_step
                else:
                        p.intermed_max_step = current_step

                #ssii_stop_at_n must be a multiple of ssii_every_n
                if not hasattr(p, 'intermed_ssii_stop_at_n'):
                    if ssii_stop_at_n % ssii_every_n == 0:
                        p.intermed_ssii_stop_at_n = ssii_stop_at_n
                    else:
                        p.intermed_ssii_stop_at_n = int(ssii_stop_at_n / ssii_every_n) * ssii_every_n

                if current_step % ssii_every_n == 0:
                    for index in range(0, p.batch_size):
                        if ssii_intermediate_type == "Denoised":
                            image = sample_to_image(d["denoised"], index=index)
                        else:
                            image = sample_to_image(d["x"], index=index)

                        logger.debug(f"ssii_intermediate_type, ssii_every_n, ssii_stop_at_n: {ssii_intermediate_type}, {ssii_every_n}, {ssii_stop_at_n}")
                        logger.debug(f"Step: {current_step}")

                        # Inits per seed
                        if current_step == 0 and p.intermed_first_pass:
                            if opts.save_images_add_number:
                                digits = 5
                            else:
                                digits = 6
                            if index == 0:
                                # Get output-dir-infos
                                fullfn = Script.save_image_only_get_name(image, p.outpath_samples, "", int(p.seed), p.prompt, p=p)
                                base_name, _ = os.path.splitext(fullfn)
                                # Set custom folder for saving intermediates on first step of first image
                                full_outpath = os.path.dirname(base_name)
                                intermed_path = os.path.join(full_outpath, "intermediates")
                                os.makedirs(intermed_path, exist_ok=True)
                                # Set filename with pattern. Two versions depending on opts.save_images_add_number
                                base_name = os.path.basename(base_name)
                                substrings = base_name.split('-')
                                if opts.save_images_add_number:
                                    intermed_number = substrings[0]
                                    intermed_number = str(intermed_number).zfill(digits)
                                    intermed_suffix = '-'.join(substrings[1:])
                                else:
                                    intermed_number = get_next_sequence_number(intermed_path, "")
                                    intermed_number = str(intermed_number).zfill(digits)
                                    intermed_suffix = '-'.join(substrings[0:])
                                intermed_path = os.path.join(intermed_path, intermed_number)
                                p.intermed_outpath = intermed_path
                                p.intermed_outpath_number = []
                                p.intermed_outpath_number.append(intermed_number)
                                p.intermed_outpath_suffix = intermed_suffix
                                # For video logic
                                p.intermed_files = []
                            else:
                                intermed_number = int(p.intermed_outpath_number[0]) + index
                                intermed_number = str(intermed_number).zfill(digits)
                                p.intermed_outpath_number.append(intermed_number)
                            logger.debug(f"p.intermed_outpath: {p.intermed_outpath}")
                            match = re.search(r"^\d+", p.intermed_outpath_suffix)
                            if match:
                                match_num = match.group()
                            else:
                                match_num = ""
                            logger.debug(f"p.intermed_outpath_suffix: {match_num}")
                            logger.debug(f"p.steps: {p.steps}")
                            logger.debug(f"p.all_seeds: {p.all_seeds}")
                            logger.debug(f"p.cfg_scale: {p.cfg_scale}")
                            logger.debug(f"p.sampler_name: {p.sampler_name}")
                            logger.debug(f"p.batch_size: {p.batch_size}")

                        intermed_suffix = p.intermed_outpath_suffix.replace(str(int(p.seed)), str(int(p.all_seeds[index])), 1)
                        p.intermed_pattern = p.intermed_outpath_number[index] + "-%%%-" + intermed_suffix
                        if hr:
                            if p.intermed_final_pass:
                                p.intermed_pattern = p.intermed_pattern.replace("%%%", "%%%-p2")
                            else:
                                p.intermed_pattern = p.intermed_pattern.replace("%%%", "%%%-p1")
                        filename = p.intermed_pattern.replace("%%%", f"{current_step:03}")

                        #don't save first step
                        if current_step > 0:
                            #generate png-info
                            infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments=[], position_in_batch=index % p.batch_size, iteration=index // p.batch_size)
                            infotext = f'{infotext}, intermediate: {current_step:03d}'

                            if current_step == p.intermed_ssii_stop_at_n:
                                if (hr and p.intermed_final_pass) or not hr:
                                    #early stop for this seed reached, prevent normal save, save as final image
                                    p.do_not_save_samples = True
                                    save_image(image, p.outpath_samples, "", p.all_seeds[index], p.prompt, opts.samples_format, info=infotext, p=p)
                                    if index == p.batch_size - 1:
                                        #early stop for final seed and final pass reached, interrupt further processing
                                        state.interrupt()
                                else:
                                    #save intermediate image
                                    save_image(image, p.intermed_outpath, "", info=infotext, p=p, forced_filename=filename, save_to_dirs=False)
                                    filename_clean = re.sub(r"[^\d-]", "%", filename)
                                    logger.debug(f"filename: {filename_clean}")
                                    if ssii_video and ((hr and p.intermed_first_pass and ssii_video_hires == "1") or (hr and p.intermed_final_pass and ssii_video_hires == "2") or not hr):
                                        p.intermed_files.append((filename + ".png", None))
                            else:
                                #save intermediate image
                                save_image(image, p.intermed_outpath, "", info=infotext, p=p, forced_filename=filename, save_to_dirs=False)
                                filename_clean = re.sub(r"[^\d-]", "%", filename)
                                logger.debug(f"filename: {filename_clean}")
                                if ssii_video and ((hr and p.intermed_first_pass and ssii_video_hires == "1") or (hr and p.intermed_final_pass and ssii_video_hires == "2") or not hr):
                                    p.intermed_files.append((filename + ".png", None))
                return orig_callback_state(self, d)

            setattr(KDiffusionSampler, "callback_state", callback_state)

    def postprocess(self, p, processed, ssii_is_active, ssii_intermediate_type, ssii_every_n, ssii_stop_at_n, ssii_video, ssii_video_format, ssii_video_fps, ssii_video_hires, ssii_debug):
        setattr(KDiffusionSampler, "callback_state", orig_callback_state)

        # Make a video file
        if ssii_is_active and ssii_video and not state.skipped and not state.interrupted:
            logger = logging.getLogger(__name__)
            # ffmpeg requires sequential numbers in filenames (that is exactly +1)
            for i, (name_org, _) in enumerate(p.intermed_files):
                num_seq = '{:03}'.format(i)
                name_seq = re.sub(r'^\d+-(\d{3})', f'{name_org.split("-")[0]}-{num_seq}', name_org)
                p.intermed_files[i] = (name_org, name_seq)
                path_name_org = os.path.join(p.intermed_outpath, name_org)
                path_name_seq = os.path.join(p.intermed_outpath, name_seq)
                os.replace(path_name_org, path_name_seq)
                logger.debug(f"replace {path_name_org} / {path_name_seq}")

            img_file = p.intermed_pattern.replace("%%%", "%03d") + ".png"
            vid_file = p.intermed_pattern.replace("%%%-", "") + "." + ssii_video_format
            if hasattr(p, "enable_hr"):
                if p.enable_hr and ssii_video_hires == "1":
                    img_file = img_file.replace("-p2-", "-p1-")
                    vid_file = vid_file.replace("-p2-", "-p1-")
            path_img_file = os.path.join(p.intermed_outpath, img_file) 
            path_vid_file = os.path.join(p.intermed_outpath, vid_file) 
            if ssii_video_format == "gif":
                ff = FFmpeg(
                    inputs={path_img_file: f"-framerate {int(ssii_video_fps)}"},
                    outputs={path_vid_file: '-filter_complex "split[v1][v2]; [v1]palettegen=stats_mode=full [palette]; [v2][palette]paletteuse=dither=sierra2_4a"'}
                )
            else:
                ff = FFmpeg(
                    inputs={path_img_file: f"-framerate {int(ssii_video_fps)}"},
                    outputs={path_vid_file: None}
                )
            ff.run()
            
            # Back to original numbering
            for i, (name_org, name_seq) in enumerate(reversed(p.intermed_files)):
                path_name_org = os.path.join(p.intermed_outpath, name_org)
                path_name_seq = os.path.join(p.intermed_outpath, name_seq)
                os.replace(path_name_seq, path_name_org)
                logger.debug(f"replace {path_name_seq} / {path_name_org}")