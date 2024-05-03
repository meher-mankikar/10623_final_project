# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Old Demo File that runs with baseline LION DDPM
"""
import os
import clip
import torch
import time
from PIL import Image

from default_config import cfg as config
from models.lion import LION
from models.lion_dpm import LionDPM
from utils.vis_helper import plot_points
from huggingface_hub import hf_hub_download 

model_path= './lion_ckpt/text2shape/car/checkpoints/model.pt'
model_config = './lion_ckpt/text2shape/car/cfg.yml'

#model_path= './lion_ckpt/unconditional/chair/checkpoints/model.pt'
#model_config = './config/chair_prior_cfg.yml'

#model_path= './lion_ckpt/unconditional/car/checkpoints/model.pt'
#model_config = './config/car_prior_cfg.yml'

object_type = model_path.split("/")[3]
config.merge_from_file(model_config)
lion = LION(config)
lion.load_model(model_path)

if config.clipforge.enable:
    input_t = ["a swivel chair, five wheels"] 
    device_str = 'cuda'
    clip_model, clip_preprocess = clip.load(
                        config.clipforge.clip_model, device=device_str)    
    text = clip.tokenize(input_t).to(device_str)
    clip_feat = []
    clip_feat.append(clip_model.encode_text(text).float())
    clip_feat = torch.cat(clip_feat, dim=0)
    print('clip_feat', clip_feat.shape)
else:
    clip_feat = None

t_start = time.time()
for i in range(1, 5):
    print(f"Starting sample {i}")
    output = lion.sample(1 if clip_feat is None else clip_feat.shape[0], clip_feat=clip_feat)
    pts = output['points']
    img_name = f'results/object_{object_type}_lion_e{i}.png'
    plot_points(pts, output_name=img_name)
    #img = Image.open(img_name)
    #img.show()
t_end = time.time()
print(f"1000 samples using {config.ddpm.dpm_step} steps: {t_end - t_start} seconds")

