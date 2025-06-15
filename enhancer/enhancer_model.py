import cv2 
import torch
import numpy as np
import torch.nn.functional as F
from enhancer.basicsr.models import create_model
from enhancer.basicsr.utils.options import dict2str, parse
import torch.nn as nn
import glob
import os

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

def enhance_image(frame, model_restoration, factor = 4, self_assemble = True):
    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img = np.float32(frame) / 255.0 
        input_ = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to('cpu')

        # Padding in case the image is not divisible by 4 
        b, c, h, w = input_.shape 
        H, W = (h + factor) // factor * factor, (w + factor) // factor * factor
        padh = H - h if h % factor != 0 else 0 
        padw = W - w if w % factor != 0 else 0 
        input_ = F.pad(input_, (0, padw, 0, padh), mode='reflect')

        if h < 3000 and w < 3000: 
            if self_assemble: 
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)

        else: 
            # SPLIT 
            input_1 = input_[:, :, :, 1::2]
            input_2 = input_[:, :, :, 0::2]
            if self_assemble: 
                restored_1 = self_ensemble(input_1, model_restoration)
                restored_2 = self_ensemble(input_2, model_restoration)
            else:
                restored_1 = model_restoration(input_1)
                restored_2 = model_restoration(input_2)
            restored = torch.zeros_like(input_)
            restored[:, :, :, 1::2] = restored_1 
            restored[:, :, :, 0::2] = restored_2 

        # Unpad the images to the original size 
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).numpy()
        restored = (restored * 255.0).astype(np.uint8)
        cv2.imwrite('enhanced_image.png', restored[0])  # Save the enhanced image for debugging 



def load_enhancer():
    yaml_path = 'enhancer/Options/RetinexFormer_LOL_v1.yml'
    opt = parse(yaml_path, is_train=False)
    opt['dist'] = False
    model_restoration = create_model(opt).net_g 

    weights = 'enhancer/experiments/RetinexFormer_LOL_v1/best_psnr_23.06_93000.pth'
    checkpoint = torch.load(weights, weights_only=True)

    try: 
        model_restoration.load_state_dict(checkpoint['params'], strict = True)
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)

    model_restoration = model_restoration.cpu()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    print("===>Loaded enhancer model with weights: ", weights)
    return model_restoration
                



