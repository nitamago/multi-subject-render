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

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Script(scripts.Script):
    def title(self):
        return "Multi Subject Rendering"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        if is_img2img: return
        
        return []

    def run(self,p):
        initial_CLIP = opts.data["CLIP_stop_at_last_layers"]
        sdmg = module_from_file("simple_depthmap",'extensions/multi-subject-render/scripts/simple_depthmap.py')
        sdmg = sdmg.SimpleDepthMapGenerator() #import midas

        fix_seed(p)
        p.do_not_save_samples = True

        foregen_size_x = p.width  if foregen_size_x == 64 else foregen_size_x
        foregen_size_y = p.height if foregen_size_y == 64 else foregen_size_y
        foregen_blend_size_x = p.width  if foregen_blend_size_x == 64 else foregen_blend_size_x
        foregen_blend_size_y = p.height if foregen_blend_size_y == 64 else foregen_blend_size_y

        o_sampler_name = p.sampler_name
        o_prompt    = p.prompt
        o_cfg_scale = p.cfg_scale
        o_steps     = p.steps
        o_do_not_save_samples = p.do_not_save_samples
        o_width     = p.width
        o_height    = p.height
        o_denoising_strength = p.denoising_strength
        o_firstphase_width   = p.firstphase_width
        o_firstphase_height  = p.firstphase_height

        n_iter=p.n_iter
        image_list = []
        for j in range(n_iter):
            p.n_iter=1
            p.prompt = o_prompt
            p.sampler_name = o_sampler_name
            p.cfg_scale = o_cfg_scale
            p.steps = o_steps
            p.do_not_save_samples = o_do_not_save_samples
            p.width = o_width
            p.height = o_height
            p.denoising_strength = o_denoising_strength
            p.firstphase_width = o_firstphase_width
            p.firstphase_height = o_firstphase_height
            proc = process_images(p)
            
            image_list.append(proc.images[0])
            
            foreground_image_mask = sdmg.calculate_depth_map_for_waifus(proc.images[0])
            image_list.append(foreground_image_mask)

        proc.images = image_list
        return proc
