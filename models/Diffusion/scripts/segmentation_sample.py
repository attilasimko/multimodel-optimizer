"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
#from visdom import Visdom
#viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torchvision.transforms as T
import torch.distributed as dist
from guided_diffusion import dist_util, logger
#from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.img_dataset import ImgDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from tqdm.auto import tqdm
from PIL import Image
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
transform = T.ToPILImage()

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args.gpu)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = ImgDataset(args.data_dir, split_flag='val')
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("sampling...") #7340
    #j = 0
    indicies = range(args.num_samples)
    for j in indicies:
    #while j < args.num_samples:
        b, path = next(data)
        if j not in [30, 100]:
            continue
        #should return an image from the dataloader "data"
        
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)  #add a noise channel

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)


        for i in tqdm(range(args.num_ensemble)):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 2, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                progress=False,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            # print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            if i == 0:
                s = th.tensor(sample)
            else:
                s = th.cat((s, sample), 0)
            
        th.save(s, './'+'_ensemble'+str(j))

            # s = (sample*255).cpu().numpy()            
            # im = Image.fromarray(s[0,0,:,:])         
            # im = im.convert("L")
            # im.save(os.path.join("./fid/sampled", str(j) + ".png"))   
            #viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            # th.save(s, './fid_outputs/'+'_output'+str(j)) #save the generated mask
        #j += 1

def create_argparser():
    defaults = dict(
        image_size=256,
        diffusion_steps = 1000,
        data_dir="/home/lorenzo/data/interim",
        clip_denoised=False, # True
        learn_sigma = True,
        num_res_blocks = 2,
        num_heads = 1,
        microbatch= -1,
        use_fp16=False,
        rescale_timesteps = False,
        rescale_learned_sigmas = False,
        fp16_scale_growth=1e-3,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./results/1220_nocond_savedmodel100000.pt",
        num_ensemble=1,      #number of samples in the ensemble
        gpu=2,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
