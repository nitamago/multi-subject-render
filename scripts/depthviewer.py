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
        return "Depth viewer"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        if is_img2img: return
        
        return []

    def run(self,p):
        initial_CLIP = opts.data["CLIP_stop_at_last_layers"]
        sdmg = module_from_file("simple_depthmap",'extensions/multi-subject-render/scripts/simple_depthmap.py')
        sdmg = sdmg.SimpleDepthMapGenerator() #import midas

        def cut_depth_mask(img,mask_img,foregen_treshold):
            img = img.convert("RGBA")
            mask_img = mask_img.convert("RGBA")
            mask_datas = mask_img.getdata()
            datas = img.getdata()
            treshold = foregen_treshold
            newData = []
            for i in range(len(mask_datas)):
                if mask_datas[i][0] >= foregen_treshold and mask_datas[i][1] >= foregen_treshold and mask_datas[i][2] >= foregen_treshold:
                    newData.append(datas[i])
                else:
                    newData.append((255, 255, 255, 0))
            mask_img.putdata(newData)
            return mask_img

        fix_seed(p)
        p.do_not_save_samples = True

        n_iter=p.n_iter
        image_list = []
        for j in range(n_iter):
            p.n_iter=1
            proc = process_images(p)
            
            foreground_image = proc.images[0]
            image_list.append(foreground_image)
            
            foreground_image_mask = sdmg.calculate_depth_map_for_waifus(foreground_image)
            image_list.append(foreground_image_mask)

            for i in range(1, 11):
                treshold = 255  * i / 10
                foreground_image = cut_depth_mask(foreground_image, foreground_image_mask, treshold)
                image_list.append(foreground_image)

        proc.images = image_list
        return proc