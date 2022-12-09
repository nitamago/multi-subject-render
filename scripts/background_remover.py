import torch
import cv2
import requests
import os.path
import contextlib
from PIL import Image
from modules.shared import opts, cmd_opts
from modules import processing, images, shared

from repositories.carvekit.api.high import HiInterface

import numpy as np

class BackGroundRemover(object):
    def remove_background(self, images):
        # Check doc strings for more information
        interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                                batch_size_seg=5,
                                batch_size_matting=1,
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                #seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                                seg_mask_size=768,  # Use 640 for Tracer B7 and 320 for U2Net
                                matting_mask_size=2048,
                                trimap_prob_threshold=231,
                                #trimap_prob_threshold=100,
                                #trimap_dilation=30,
                                trimap_dilation=5,
                                trimap_erosion_iters=5,
                                fp16=False)
        images_without_background = interface(images)
        return images_without_background