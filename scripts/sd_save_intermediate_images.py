import datetime
import json
import locale
import logging
import os
import platform
import re
import shutil
import sys

from modules import paths
from modules import scripts
from modules import script_callbacks
from modules.processing import Processed, process_images, fix_seed, create_infotext
try:
    from modules.sd_samplers_kdiffusion import KDiffusionSampler
    from modules.sd_samplers_common import sample_to_image
except ImportError:
    from modules.sd_samplers import KDiffusionSampler, sample_to_image
from modules.images import save_image, FilenameGenerator, get_next_sequence_number
from modules.shared import opts, state, cmd_opts

from ffmpy import FFmpeg
import gradio as gr; gr.__version__

# New args, regex for npp, find: (ssii_hires)(?=,)
# replace: \1, ssii_add_last_frames, ssii_add_first_frames
# plus debug

orig_callback_state = KDiffusionSampler.callback_state
ui_config_backup = os.path.join(scripts.basedir(), "ui-config_backup.json")
video_bat_mode = ""
ui_items = {
    "ssii_is_active": "Save intermediate images",
    "ssii_final_save": "Also save final image with intermediates",
    "ssii_save_settings": "Save current settings as default",
    "ssii_intermediate_type": "Type of images to be saved",
    "ssii_every_n": "Save every N images",
    "ssii_start_at_n": "Start at N images (must be 0 = start at the beginning or a multiple of 'Save every N images')",
    "ssii_stop_at_n": "Stop at N images (must be 0 = don't stop early or a multiple of 'Save every N images')",
    "ssii_mode": "Mode",
    "ssii_video_format": "Format",
    "ssii_mp4_parms": "mp4 parameters",
    "ssii_video_fps": "fps",
    "ssii_add_first_frames": "Display last image for additional frames at the beginning",
    "ssii_add_last_frames": "Display last image for additional frames at the end",
    "ssii_smooth": "Smoothing / Interpolate",
    "ssii_seconds": "Approx. how many seconds should the video run?",
    "ssii_lores": "lores",
    "ssii_hires": "hires",
	"ssii_ffmpeg_bat": "Create Windows ffmpeg bat-files for lores and hires. Changes numbering logic.",
	"ssii_bat_only": "Only bat-files, no video",
    "ssii_debug": "Debug"
}
mode_choices = ["Make a video file", "Save intermediates with image number as suffix", "Standard operation"]

def ui_setting_set(ui_settings, key, value):
    this_module = os.path.basename(__file__)
    ui_settings[f"customscript/{this_module}/txt2img/{key}/value"] = value
    ui_settings[f"customscript/{this_module}/img2img/{key}/value"] = value
    return ui_settings

def ssii_save_settings_do(ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug):
    ui_config_file = cmd_opts.ui_config_file
    ui_settings = {}

    try:
        if os.path.exists(ui_config_file):
            with open(ui_config_file, "r", encoding="utf8") as file:
                ui_settings = json.load(file)
    except Exception:
        message = f"Error loading settings: {sys.stderr}"
        return message

    # Save one backup
    with open(ui_config_backup, "w", encoding="utf8") as file:
        json.dump(ui_settings, file, indent=4)

    for key, value in ui_items.items():
        if key != "ssii_is_active" and key != "ssii_save_settings":
            ui_settings = ui_setting_set(ui_settings, value, eval(key))

    with open(ui_config_file, "w", encoding="utf8") as file:
        json.dump(ui_settings, file, indent=4)

    locale.setlocale(locale.LC_ALL, '')
    current_date_and_time = datetime.datetime.now()
    message = f"Last saved as default: {current_date_and_time.strftime('%c')}"

    return f"<p align = right>{message}</p>"

def get_add_num():
    if video_bat_mode == "video":
        add_num = 1000
    elif video_bat_mode == "bat_lores":
        add_num = 2000
    elif video_bat_mode == "bat_hires":
        add_num = 3000

    return add_num

def ssii_set_num(name, i):
    add_num = get_add_num()
    if i < add_num:
        i = i + add_num
    num = '{:04}'.format(i)
    name_set_num = re.sub(r'^\d+-(\d{3,4})', f'{name.split("-")[0]}-{num}', name)

    return name_set_num

def ssii_add_last_frames_add(add_frames, add_files, i, batch_no, name_org, name_seq):
    for frame in range(int(add_frames)):
        name_added = ssii_set_num(name_seq, i)
        add_files.append((batch_no, name_org, f"link:{name_added}"))
        i = i + 1
    return add_files, i

def ssii_add_last_frames_logic(p, ssii_add_first_frames, ssii_add_last_frames, ssii_debug):
    if ssii_add_first_frames > 0 or ssii_add_last_frames > 0:
        add_files = []
        prev_batch = None
        for real_i, (batch_no, name_org, name_seq) in enumerate(p.intermed_files):
            if prev_batch != batch_no:
                if ssii_add_last_frames > 0 and prev_batch is not None:
                    add_files, i = ssii_add_last_frames_add(ssii_add_last_frames, add_files, i, prev_batch, name_changed, prev_name_seq)

                i = 0

                if ssii_add_first_frames > 0:
                    name_last = next(c for a,b,c in p.intermed_files[::-1] if a == batch_no)
                    match = re.search(r"^\d+-(\d{4})", name_last)
                    num_last = int(match.group(1)) + int(ssii_add_first_frames)
                    name_last = ssii_set_num(name_last, num_last)
                    add_files, i = ssii_add_last_frames_add(ssii_add_first_frames, add_files, i, batch_no, name_last, name_seq)

            name_changed = ssii_set_num(name_seq, i)
            add_files.append((batch_no, name_org, name_changed))
            i = i + 1
            prev_batch = batch_no
            prev_name_seq = name_seq

        if ssii_add_last_frames > 0:
            add_files, i = ssii_add_last_frames_add(ssii_add_last_frames, add_files, i, batch_no, name_changed, name_seq)

        p.intermed_files = add_files
    return

def ffmpeg(inputs=None, outputs=None):
    ff_ffmpeg = FFmpeg(inputs=inputs, outputs=outputs)

    input_key = next(iter(inputs))
    input_value = next(iter(inputs.values()))
    output_key = next(iter(outputs))
    output_value = next(iter(outputs.values()))
    
    input_key = os.path.normpath(input_key)
    input_key = input_key.replace("%04d", "%%04d", 1)
    output_key = os.path.normpath(output_key)
    
    ff_bat = f'ffmpeg.exe {input_value} -i "{input_key}" {output_value} "{output_key}"'

    return (ff_ffmpeg, ff_bat)

def make_video(p, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug):
    global video_bat_mode
    if ssii_is_active and ssii_mode == mode_choices[0] and ((not state.skipped and not state.interrupted) or p.intermed_stopped):
        do_video = True
        do_bat = False
        if ssii_ffmpeg_bat and hr_check(p):
            do_bat = True
            if ssii_bat_only:
                do_video = False

        intermed_files_store = p.intermed_files
        if do_video:
            video_bat_mode = "video"
            make_video_or_bat(p, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug)
        if do_bat:
            p.intermed_files = intermed_files_store
            video_bat_mode = "bat_lores"
            make_video_or_bat(p, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug)
            p.intermed_files = intermed_files_store
            video_bat_mode = "bat_hires"
            make_video_or_bat(p, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug)

    return

def link_file(path_name_org, path_name_seq, hard_links):
    if hard_links:
        os.link(path_name_org, path_name_seq)
    else:
        shutil.copyfile(path_name_org, path_name_seq)
    
    return

def make_video_or_bat(p, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug):
    logger = logging.getLogger(__name__)
    logger.debug(f"video mode: {video_bat_mode}")
    if video_bat_mode == "video":
        lores = ssii_lores
        hires = ssii_hires
    elif video_bat_mode == "bat_lores":
        lores = True
        hires = False
    elif video_bat_mode == "bat_hires":
        lores = False
        hires = True
    add_num = get_add_num()

    # remove unwanted files according to lores/hires setting
    if hr_check(p) and (not lores or not hires):
        intermed_files_remain = []
        for (batch_no, name_org, name_seq) in p.intermed_files:
            filename = os.path.basename(name_org)
            parts = filename.split("-")
            current_step = parts[1]
            if (lores and int(current_step) < int(p.intermed_hires_start)) or (hires and int(current_step) >= int(p.intermed_hires_start)):
                    intermed_files_remain.append((batch_no, name_org, name_seq))
                    logger.debug(f"Not removed: {filename_clean(name_org)}")
        p.intermed_files = intermed_files_remain

    # ffmpeg requires sequential numbers in filenames (that is exactly +1)
    p.intermed_files.sort(key=lambda x: x[0])
    prev_batch = None
    for real_i, (batch_no, name_org, _) in enumerate(p.intermed_files):
        if prev_batch != batch_no:
            i = 0
        name_seq = ssii_set_num(name_org, i)
        p.intermed_files[real_i] = (batch_no, name_org, name_seq)
        i = i + 1
        prev_batch = batch_no
        frames_per_image = i

    if ssii_add_first_frames > 0 or ssii_add_last_frames > 0:
        ssii_add_last_frames_logic(p, ssii_add_first_frames, ssii_add_last_frames, ssii_debug)

    p.intermed_files.sort(key=lambda x: (x[0], x[2].startswith("link:")))

    hard_links = True
    for (batch_no, name_org, name_seq) in p.intermed_files:
        path_name_org = os.path.join(p.intermed_outpath, name_org)
        if name_seq[:5] == "link:":
            path_name_seq = os.path.join(p.intermed_outpath, name_seq[5:])
            try:
                link_file(path_name_org, path_name_seq, hard_links)
            except OSError as e:
                if e.errno == 38 and hard_links:
                    hard_links = False
                    link_file(path_name_org, path_name_seq, hard_links)
                else:
                    raise
        else:
            path_name_seq = os.path.join(p.intermed_outpath, name_seq)
            os.replace(path_name_org, path_name_seq)
        logger.debug(f"replace/link {filename_clean(path_name_org)} / {filename_clean(path_name_seq)}")

    for intermed_pattern in p.intermed_pattern.values():
        img_file = intermed_pattern.replace("%%%", "%04d") + ".png"
        path_img_file = os.path.join(p.intermed_outpath, img_file) 
        if video_bat_mode == "video":
            vid_file = intermed_pattern.replace("%%%-", "") + "." + ssii_video_format
            path_vid_file = os.path.join(p.intermed_outpath, vid_file)
        else:
            path_img_file = os.path.abspath(path_img_file)
            if video_bat_mode == "bat_lores":
                vid_file = intermed_pattern.replace("%%%", "lo") + "." + ssii_video_format
            else:
                vid_file = intermed_pattern.replace("%%%", "hi") + "." + ssii_video_format
            path_vid_file = os.path.abspath(os.path.join(p.intermed_outpath, vid_file))
         
        if ssii_video_format == "mp4":
            if ssii_mp4_parms == "h265/hevc":
                mp4_parms = " -c:v libx265 -vtag hvc1"
            elif ssii_mp4_parms == "av1":    
                mp4_parms = " -c:v librav1e -speed 4 -tile-columns 2 -tile-rows 2"
            else:
                mp4_parms = " -c:v libx264 -profile:v high -pix_fmt yuv420p"
        else:
            mp4_parms = ""
        if ssii_smooth:
            pts = (round(ssii_seconds / frames_per_image, 5))
            logger.debug(f"pts: {pts}")
            if pts < 1:
                pts = "1"
            else:
                pts = str(pts)
            if ssii_video_format == "gif":
                ff = ffmpeg(
                    inputs={path_img_file: f"-benchmark -framerate 1 -start_number {add_num}"},
                    outputs={path_vid_file: f'-filter_complex "split[v1][v2]; [v1]palettegen=stats_mode=full [palette]; [v2][palette]paletteuse=dither=sierra2_4a [v3]; [v3]setpts={pts}*PTS [v4]; [v4]minterpolate=fps={int(ssii_video_fps)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"'}
                )
            else:
                ff = ffmpeg(
                    inputs={path_img_file: f"-benchmark -framerate 1 -start_number {add_num}"},
                    outputs={path_vid_file: f'-filter_complex "setpts={pts}*PTS [v4]; [v4]minterpolate=fps={int(ssii_video_fps)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" -bufsize 2000k {mp4_parms}'}
                )
        else:
            if ssii_video_format == "gif":
                ff = ffmpeg(
                    inputs={path_img_file: f"-benchmark -framerate {int(ssii_video_fps)} -start_number {add_num}"},
                    outputs={path_vid_file: '-filter_complex "split[v1][v2]; [v1]palettegen=stats_mode=full [palette]; [v2][palette]paletteuse=dither=sierra2_4a"'}
                )
            else:
                ff = ffmpeg(
                    inputs={path_img_file: f"-benchmark -framerate {int(ssii_video_fps)} -start_number {add_num}"},
                    outputs={path_vid_file: f"-bufsize 2000k{mp4_parms}"}
                )
        if video_bat_mode == "video":
            ff_fmpeg = ff[0]
            ff_fmpeg.run()
        else:
            bat_filename = os.path.splitext(path_vid_file)[0] + ".bat"
            with open(bat_filename, "w") as file:
                file.write(ff[1])
    
    if video_bat_mode == "video":
        # Back to original numbering
        for (batch_no, name_org, name_seq) in p.intermed_files:
            path_name_org = os.path.join(p.intermed_outpath, name_org)
            if name_seq[:5] == "link:":
                path_name_seq = os.path.join(p.intermed_outpath, name_seq[5:])
                os.unlink(path_name_seq)
            else:
                path_name_seq = os.path.join(p.intermed_outpath, name_seq)
                os.replace(path_name_seq, path_name_org)
            logger.debug(f"replace/unlink {filename_clean(path_name_seq)} / {filename_clean(path_name_org)}")
    return

def filename_clean(filename):
    filename_cleaned = re.sub(r"[^\d/\\-]", "%", filename)
    return filename_cleaned

def hr_check(p):
    if hasattr(p, "enable_hr"):
        hr = p.enable_hr
    else:
        hr = False 
    return hr
    
def hr_active_check(p):
    if hasattr(p, "intermed_final_pass"):
        hr_active = p.intermed_final_pass
    else:
        hr_active = False
    return hr_active

class Script(scripts.Script):
    def title(self):
        return "Save intermediate images during the sampling process"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Save intermediate images", open=False):
            with gr.Row():
                ssii_message = gr.HTML()
            with gr.Row():
                ssii_is_active = gr.Checkbox(
                    label=ui_items["ssii_is_active"],
                    value=False
                )
                ssii_final_save = gr.Checkbox(
                    label=ui_items["ssii_final_save"],
                    value=False
                )
                ssii_save_settings = gr.Button(
                    value=ui_items["ssii_save_settings"],
                    elem_id="ssii_save_settings"
                    )
            with gr.Row():
                ssii_intermediate_type = gr.Radio(
                    label=ui_items["ssii_intermediate_type"],
                    choices=["Denoised", "Noisy", "According to Live preview subject setting"],
                    value="Denoised"
                )
            with gr.Row():
                ssii_every_n = gr.Number(
                    label=ui_items["ssii_every_n"],
                    value="5"
                )
            with gr.Row():
                ssii_start_at_n = gr.Number(
                    label=ui_items["ssii_start_at_n"],
                    value="0"
                )
            with gr.Row():
                ssii_stop_at_n = gr.Number(
                    label=ui_items["ssii_stop_at_n"],
                    value="0"
                )
            with gr.Box():
                with gr.Row():
                    ssii_mode = gr.Radio(
                        label=ui_items["ssii_mode"],
                        choices=mode_choices,
                        value=mode_choices[2]
                    )
                with gr.Row():
                    with gr.Box():
                        ssii_video_format = gr.Radio(
                            label=ui_items["ssii_video_format"],
                            choices=["gif", "webm", "mp4"],
                            value="mp4"
                        )
                        ssii_mp4_parms = gr.Radio(
                            label=ui_items["ssii_mp4_parms"],
                            choices=["h264", "h265/hevc", "av1"],
                            value="h264"
                        )
                    ssii_video_fps = gr.Number(
                        label=ui_items["ssii_video_fps"],
                        value=2
                    )
                with gr.Box():
                    with gr.Row():
                        ssii_add_first_frames = gr.Number(
                            label=ui_items["ssii_add_first_frames"],
                            value=0
                        )
                        ssii_add_last_frames = gr.Number(
                            label=ui_items["ssii_add_last_frames"],
                            value=0
                        )
                with gr.Box():
                    with gr.Row():
                        ssii_smooth = gr.Checkbox(
                            label=ui_items["ssii_smooth"],
                            value=False
                        )
                        
                        ssii_seconds = gr.Number(
                            label=ui_items["ssii_seconds"],
                            value=0
                        )
                    with gr.Row():
                        gr.HTML("fps >= 30 recommended, caution: generates large gif-files")
                with gr.Box():
                    with gr.Row():
                        ssii_lores = gr.Checkbox(
                            label=ui_items["ssii_lores"],
                            value=True
                        )
                        ssii_hires = gr.Checkbox(
                            label=ui_items["ssii_hires"],
                            value=True
                        )
                    with gr.Row():
                        ssii_ffmpeg_bat = gr.Checkbox(
                            label=ui_items["ssii_ffmpeg_bat"],
                            value=False
                        )
                        ssii_bat_only = gr.Checkbox(
                            label=ui_items["ssii_bat_only"],
                            value=False
                        )
            with gr.Row():
                ssii_debug = gr.Checkbox(
                    label=ui_items["ssii_debug"],
                    value=False
                )

        ssii_save_settings.click(
            fn=ssii_save_settings_do,
            inputs=[ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug],
            outputs=[ssii_message],
        )

        return [ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug]

    def save_image_only_get_name(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
        # for description see modules.images.save_image, same code up saving of files
        
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

    def process(self, p, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug):
        if ssii_is_active:

            # Debug logging
            if ssii_debug:
                mode = logging.DEBUG
            else:
                mode = logging.WARNING
            logger = logging.getLogger(__name__)
            logger.setLevel(mode)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(mode)
            logger.addHandler(console_handler)
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

            p.intermed_stopped = False
            p.intermed_is_active = ssii_is_active
            p.intermed_final_save = ssii_final_save
            p.intermed_video = ssii_mode
            
            def callback_state(self, d):
                """
                callback_state runs after each processing step
                """
                current_step = d["i"]

                hr = hr_check(p)

                # Highres. fix requires 2 passes
                if not hasattr(p, 'intermed_final_pass'):
                    if hr:
                        p.intermed_first_pass = True
                        p.intermed_final_pass = False
                    else:
                        p.intermed_first_pass = True
                        p.intermed_final_pass = True

                # Check if pass 1 has finished
                if hasattr(p, 'intermed_max_step'):
                    if current_step >= p.intermed_max_step:
                        p.intermed_max_step = current_step
                    else:
                        p.intermed_first_pass = False
                        p.intermed_final_pass = True
                        p.intermed_max_step = current_step
                else:
                        p.intermed_max_step = current_step

                if current_step == 0:
                    # Deal with batch_count > 1
                    if hasattr(p, 'intermed_batch_iter'):
                        if p.iteration > p.intermed_batch_iter:
                            p.intermed_batch_iter = p.iteration
                            # Reset per-batch_count-attributes
                            delattr(p, "intermed_final_pass")
                            delattr(p, "intermed_max_step")
                            if hr:
                                delattr(p, "intermed_hires_start")
                            # Make video for previous batch_count
                            make_video(p, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug)
                    else:
                        p.intermed_batch_iter = p.iteration

                abs_step = current_step
                hr_active = False
                if hr:
                    hr_active = hr_active_check(p)
                    if hr_active:
                        abs_step = current_step + p.steps
                        if not hasattr(p, 'intermed_hires_start'):
                            p.intermed_hires_start = abs_step

                logger.debug("ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug:")
                logger.debug(f"{ssii_intermediate_type}, {ssii_every_n}, {ssii_start_at_n}, {ssii_stop_at_n}, {ssii_mode}, {ssii_video_format}, {ssii_mp4_parms}, {ssii_video_fps}, {ssii_add_first_frames}, {ssii_add_last_frames}, {ssii_smooth}, {ssii_seconds}, {ssii_lores}, {ssii_hires}, {ssii_ffmpeg_bat}, {ssii_bat_only}, {ssii_debug}")
                logger.debug(f"Step, abs_step, hr, hr_active: {current_step}, {abs_step}, {hr}, {hr_active}")

                # ssii_start_at_n must be a multiple of ssii_every_n
                if not hasattr(p, 'intermed_ssii_start_at_n'):
                    if ssii_start_at_n % ssii_every_n == 0:
                        p.intermed_ssii_start_at_n = ssii_start_at_n
                    else:
                        p.intermed_ssii_start_at_n = int(ssii_start_at_n / ssii_every_n) * ssii_every_n

                # ssii_stop_at_n must be a multiple of ssii_every_n
                if not hasattr(p, 'intermed_ssii_stop_at_n'):
                    if ssii_stop_at_n % ssii_every_n == 0:
                        p.intermed_ssii_stop_at_n = ssii_stop_at_n
                    else:
                        p.intermed_ssii_stop_at_n = int(ssii_stop_at_n / ssii_every_n) * ssii_every_n

                if abs_step % ssii_every_n == 0:
                    for index in range(0, p.batch_size):
                        # Live preview only works on first batch_pos
                        if ssii_intermediate_type == "According to Live preview subject setting" and index == 0:
                            image = state.current_image
                        elif ssii_intermediate_type == "Noisy":
                            image = sample_to_image(d["x"], index=index)
                        else:
                            image = sample_to_image(d["denoised"], index=index)

                        logger.debug(f"ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n: {ssii_intermediate_type}, {ssii_every_n}, {ssii_start_at_n}, {ssii_stop_at_n}")
                        logger.debug(f"Step, abs_step, hr, hr_active: {current_step}, {abs_step}, {hr}, {hr_active}")
                        logger.debug(f"batch_count, iteration, batch_size, batch_pos: {p.n_iter}, {p.iteration}, {p.batch_size}, {index}")

                        # Inits per seed
                        if abs_step == 0:
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
                                p.intermed_last = {}
                                p.intermed_pattern = {}
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
                        
                        # Don't continue with no image (can happen with live preview subject setting)
                        if image is None:
                            logger.debug("image is None")
                        else:
                            intermed_seed_index = p.iteration * p.batch_size + index
                            intermed_seed = int(p.all_seeds[intermed_seed_index])
                            logger.debug(f"intermed_seed_index, intermed_seed: {intermed_seed_index}, {intermed_seed}")
                            intermed_suffix = p.intermed_outpath_suffix.replace(str(int(p.seed)), str(intermed_seed), 1)
                            if ssii_mode == mode_choices[1]:
                                intermed_pattern = p.intermed_outpath_number[index] + "-" + intermed_suffix + "-%%%"
                            else:
                                intermed_pattern = p.intermed_outpath_number[index] + "-%%%-" + intermed_suffix
                            p.intermed_pattern[intermed_seed] = intermed_pattern
                            filename = intermed_pattern.replace("%%%", f"{abs_step:03}")

                            # Don't save first step or if before start_at
                            if abs_step == 0 or abs_step < p.intermed_ssii_start_at_n:
                                logger.debug(f"abs_step, p.intermed_ssii_start_at_n: {abs_step}, {p.intermed_ssii_start_at_n}")
                            else:
                                # generate png-info
                                infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments=[], position_in_batch=index % p.batch_size, iteration=index // p.batch_size)
                                infotext = f'{infotext}, intermediate: {abs_step:03d}'

                                intermed_save = True
                                if abs_step == p.intermed_ssii_stop_at_n:
                                    # early stop for this seed reached, prevent normal save, save as final image
                                    p.do_not_save_samples = True
                                    save_image(image, p.outpath_samples, "", intermed_seed, p.prompt, opts.samples_format, info=infotext, p=p)
                                    if index == p.batch_size - 1:
                                        # early stop for final seed and final pass reached, interrupt further processing
                                        intermed_save = False
                                        p.intermed_stopped = True
                                        state.interrupt()
                                if intermed_save:
                                    # save intermediate image
                                    save_image(image, p.intermed_outpath, "", info=infotext, p=p, forced_filename=filename, save_to_dirs=False)
                                    logger.debug(f"filename: {filename_clean(filename)}")
                                    p.intermed_files.append((index, filename + ".png", None))
                                    p.intermed_last[index] = (filename + ".png", p.intermed_outpath, False)
                                    logger.debug(f"index, p.intermed_last[index]: {index}, {filename_clean(p.intermed_last[index][0])}, {filename_clean(p.intermed_last[index][1])}, {p.intermed_last[index][2]}")
                return orig_callback_state(self, d)

            setattr(KDiffusionSampler, "callback_state", callback_state)

    def postprocess(self, p, processed, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug):
        setattr(KDiffusionSampler, "callback_state", orig_callback_state)

        # Make video for last batch_count
        make_video(p, ssii_is_active, ssii_final_save, ssii_intermediate_type, ssii_every_n, ssii_start_at_n, ssii_stop_at_n, ssii_mode, ssii_video_format, ssii_mp4_parms, ssii_video_fps, ssii_add_first_frames, ssii_add_last_frames, ssii_smooth, ssii_seconds, ssii_lores, ssii_hires, ssii_ffmpeg_bat, ssii_bat_only, ssii_debug)

def handle_image_saved(params : script_callbacks.ImageSaveParams):
    if hasattr(params.p, "intermed_is_active"):
        # Copy final image to intermediates folder
        if params.p.intermed_is_active and params.p.intermed_final_save:
            directories = os.path.normpath(params.filename).split(os.sep)
            if "intermediates" not in directories:
                # Get last file name of current index
                last_found = False
                for index, (last_filename, last_path, last_done) in enumerate(params.p.intermed_last.values()):
                    if not last_done:
                        last_found = True
                        params.p.intermed_last[index] = (last_filename, last_path, True)
                        break

                if last_found:
                    # Convert final file name to intermediates filename
                    match1 = re.search(r'^(.*?)-(\d+)(.*)$', last_filename)
                    new_number = str(int(match1.group(2)) + 1).zfill(3)
                    file_new = match1.group(1) + '-' + new_number + match1.group(3)
                    #file_new_path = os.path.join(os.path.dirname(params.p.intermed_lastfile), file_new)
                    file_new_path = os.path.join(last_path, file_new)
                    logger = logging.getLogger(__name__)
                    logger.debug(f"last_filename: {filename_clean(last_filename)}")
                    logger.debug(f"file_new_path: {filename_clean(file_new_path)}")
                    
                    shutil.copy(params.filename, file_new_path)

                    # Add info for make video
                    params.p.intermed_files.append((index, file_new, None))
                
script_callbacks.on_image_saved(handle_image_saved)
