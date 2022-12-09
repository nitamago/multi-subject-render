from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images, fix_seed
from modules.shared import opts, cmd_opts, state
from PIL import Image, ImageOps, ImageFilter
from math import ceil
import cv2

import modules.scripts as scripts
from modules import sd_samplers
from modules.processing import process_images
from random import randint, shuffle
import random
from skimage.util import random_noise
import gradio as gr
import numpy as np
import sys
import os
import importlib.util
from PIL import Image


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Script(scripts.Script):
    def title(self):
        return "Multi Subject Compose"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        if is_img2img: return
        txt2img_samplers_names = [s.name for s in sd_samplers.samplers]
        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]

        # foreground UI
        with gr.Box():
            images      = gr.Textbox(label="Image paths  ", lines=5, max_lines=2000)
            material_parameters     = gr.Textbox(label="Material parameters  ", lines=5, max_lines=2000)

        # blend UI
        with gr.Box():
            foregen_blend_prompt             = gr.Textbox(label="final blend prompt", lines=2, max_lines=2000)
            foregen_blend_steps              = gr.Slider(minimum=1, maximum=120, step=1, label='blend steps   ', value=64)
            foregen_blend_cfg_scale          = gr.Slider(minimum=1, maximum=30, step=0.1, label='blend cfg scale  ', value=7.5)
            foregen_blend_denoising_strength = gr.Slider(minimum=0.1, maximum=1, step=0.01, label='blend denoising strength   ', value=0.42)
            foregen_blend_sampler            = gr.Dropdown(label="blend sampler", choices=img2img_samplers_names, value="DDIM")
            with gr.Row():
                foregen_blend_size_x  = gr.Slider(minimum=64, maximum=2048, step=64, label='blend width   (64 = same size as background) ', value=64)
                foregen_blend_size_y  = gr.Slider(minimum=64, maximum=2048, step=64, label='blend height  (64 = same size as background) ', value=64)

        with gr.Row():
            foregen_face_correction = gr.Checkbox(label='Face correction ', value=True)
            foregen_reverse_order = gr.Checkbox(label='Reverse order ', value=False)
        # foregen_mask_blur = gr.Slider(minimum=0, maximum=12, step=1, label='Mask blur', value=4)
        return    [images,
                    material_parameters,
                    foregen_blend_prompt,
                    foregen_blend_steps,
                    foregen_blend_cfg_scale,
                    foregen_blend_denoising_strength,
                    foregen_blend_sampler,
                    foregen_blend_size_x,
                    foregen_blend_size_y,
                    foregen_face_correction,
                    foregen_reverse_order,
                    ]

    def run(self,p,images,
                    material_parameters,
                    foregen_blend_prompt,
                    foregen_blend_steps,
                    foregen_blend_cfg_scale,
                    foregen_blend_denoising_strength,
                    foregen_blend_sampler,
                    foregen_blend_size_x,
                    foregen_blend_size_y,
                    foregen_face_correction,
                    foregen_reverse_order,
                    ):
        bgr = module_from_file("background_remover",'extensions/multi-subject-render/scripts/background_remover.py')
        bgr = bgr.BackGroundRemover()

        def paste_foreground(background,foreground,index,total_foreground,x_shift,y_shift,foregen_reverse_order):
            background = background.convert("RGBA")
            if not foregen_reverse_order:
                index = total_foreground-index-1
            image = Image.new("RGBA", background.size)
            image.paste(background, (0,0), background)
            image.paste(foreground, (x_shift,y_shift), foreground)
            return image

        fix_seed(p)
        foregen_blend_size_x = p.width  if foregen_blend_size_x == 64 else foregen_blend_size_x
        foregen_blend_size_y = p.height if foregen_blend_size_y == 64 else foregen_blend_size_y

        o_sampler_name = p.sampler_name
        o_prompt    = p.prompt
        o_cfg_scale = p.cfg_scale
        o_steps     = p.steps
        o_do_not_save_samples = p.do_not_save_samples
        o_denoising_strength = p.denoising_strength
        o_firstphase_width   = p.firstphase_width
        o_firstphase_height  = p.firstphase_height

        n_iter=p.n_iter
        for j in range(n_iter):
            if state.interrupted:
                break
            p.n_iter=1

            #background image processing
            p.prompt = o_prompt
            p.sampler_name = o_sampler_name
            p.cfg_scale = o_cfg_scale
            p.steps = o_steps
            p.do_not_save_samples = o_do_not_save_samples
            #p.width = o_width
            #p.height = o_height
            p.denoising_strength = o_denoising_strength
            p.firstphase_width = o_firstphase_width
            p.firstphase_height = o_firstphase_height
            proc = process_images(p)
            background_image = proc.images[0]

            # foregrounds processing
            image_paths = images.splitlines()
            material_parameter_lines = material_parameters.splitlines()
            assert len(image_paths) == len(material_parameter_lines), 'Image count not match Material parameters'
        
            thresholds = []
            x_shifts = []
            y_shifts = []
            for num, line in enumerate(material_parameter_lines):
                if num == 0:
                    continue
                parts = line.split('|')
                thresholds.append(int(parts[0]))
                x_shifts.append(int(parts[1]))
                y_shifts.append(int(parts[2]))

            background_image = Image.open(image_paths.pop(0))

            foregrounds = image_paths
            foreground_masked = bgr.remove_background(foregrounds)

            #stretch background to final blend if the final blend as a specific size set
            b_width, b_hight = background_image.size
            if b_width != foregen_blend_size_x or b_hight != foregen_blend_size_y :
                background_image = background_image.resize((foregen_blend_size_x, foregen_blend_size_y), Image.Resampling.LANCZOS)

            image_mask_background = Image.new(mode = "RGBA", size = (foregen_blend_size_x, foregen_blend_size_y), color = (0, 0, 0, 255))
            # cut depthmaps and stick foreground on the background
            foregen_iter = len(foreground_masked)
            random_order = [k for k in range(foregen_iter)]

            for f in range(foregen_iter):
                foreground_image = foreground_masked[f]
                # paste foregrounds onto background
                foregen_x_shift = x_shifts[f]
                foregen_y_shift = y_shifts[f]
                background_image = paste_foreground(background_image,foreground_image,random_order[f],foregen_iter,foregen_x_shift,foregen_y_shift,foregen_reverse_order)
                
            # final blend
            img2img_processing = StableDiffusionProcessingImg2Img(
                init_images=[background_image],
                resize_mode=0,
                denoising_strength=foregen_blend_denoising_strength,
                mask=None,
                mask_blur=0,
                inpainting_fill=1,
                inpaint_full_res=True,
                inpaint_full_res_padding=0,
                inpainting_mask_invert=1,
                sd_model=p.sd_model,
                outpath_samples=p.outpath_samples,
                outpath_grids=p.outpath_grids,
                prompt=foregen_blend_prompt,
                styles=p.styles,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=foregen_blend_sampler,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                steps=foregen_blend_steps,
                cfg_scale=foregen_blend_cfg_scale,
                width=foregen_blend_size_x,
                height=foregen_blend_size_y,
                restore_faces=foregen_face_correction,
                tiling=p.tiling,
                do_not_save_samples=False,
                do_not_save_grid=p.do_not_save_grid,
                extra_generation_params=p.extra_generation_params,
                overlay_images=p.overlay_images,
                negative_prompt=p.negative_prompt,
                eta=p.eta
                )
            final_blend = process_images(img2img_processing)
            p.subseed = p.subseed + 1 if p.subseed_strength  > 0 else p.subseed
            p.seed    = p.seed    + 1 if p.subseed_strength == 0 else p.seed
        return final_blend
