from . base_model import BaseModel
import argparse
from models.guided_diffusion.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            args_to_dict,
            add_dict_to_argparser
        )
from models.guided_diffusion import dist_util
from models.guided_diffusion.resample import create_named_schedule_sampler, UniformSampler
from models.guided_diffusion.train_util import TrainLoop
import torch.distributed as dist
import torch as th
import numpy as np
from tqdm.auto import tqdm


config = {
        "algorithm": "bayes",
        "name": "diffusion",
        "spec": {"maxCombo": 35, "objective": "minimize", "metric": "val_loss"},
        "parameters": {
            "lr": {"type": "float", "scalingType": "loguniform", "min": 0.00002, "max": 0.002},
            "batch_size": 6,
        },
        "trials": 1,
}

def predict_on_batch(exp, models, x):
    # x = x.squeeze()
    model = models[0]
    diffusion = models[1]

    model.to(dist_util.dev())
    model.eval()
    im_size = 256

    c = th.randn_like(x[:, :1, ...])
    img = th.cat((x, c), dim=1)  #add a noise channel

    for i in tqdm(range(1)):  #this is for the generation of an ensemble of 5 masks.
        model_kwargs = { }
        sample_fn = diffusion.p_sample_loop_known
        sample, x_noisy, org = sample_fn(
            model,
            (exp.get_parameter('batch_size'), 2, im_size, im_size), img,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
        # print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

        if i == 0:
            s = th.tensor(sample)
        else:
            s = th.cat((s, sample), 1)
                
    return np.mean(s.numpy(), 1)

class DiffusionModel(BaseModel):
    def create_argparser():
        defaults = dict(
        data_dir="/home/lorenzo/data/interim",
        img_size=256,
        schedule_sampler="uniform",
        save_prefix="XXXX",
        sample_step=None, #Sample and save an image at this step
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu=2,
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

    def build_model(exp, task):
        args = DiffusionModel.create_argparser().parse_args()
        model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())
        
        dist_util.setup_dist(exp.get_parameter("gpu"))

        exp.log_parameters(
            {
                "microbatch": args.microbatch,
                "ema_rate": args.ema_rate,
                "log_interval": args.log_interval,
                "save_interval": args.save_interval,
                "resume_checkpoint": args.resume_checkpoint,
                "use_fp16": args.use_fp16,
                "fp16_scale_growth": args.fp16_scale_growth,
                "weight_decay": args.weight_decay,
                "lr_anneal_steps": args.lr_anneal_steps,
                "save_prefix": args.save_prefix,
                "sample_step": args.sample_step, #Sample and save an image at this step
            })
        return (model, diffusion)

    def train(experiment, model, datal):
        TrainLoop(
            model=model[0],
            diffusion=model[1],
            classifier=None,
            data=None, # Data is not used, we think.
            dataloader=datal.dataloader,
            batch_size=experiment.get_parameter("batch_size"),
            microbatch=experiment.get_parameter("microbatch"),
            lr=experiment.get_parameter("lr"),
            ema_rate=experiment.get_parameter("ema_rate"),
            log_interval=experiment.get_parameter("log_interval"),
            save_interval=experiment.get_parameter("save_interval"),
            resume_checkpoint=experiment.get_parameter("resume_checkpoint"),
            use_fp16=experiment.get_parameter("use_fp16"),
            fp16_scale_growth=experiment.get_parameter("fp16_scale_growth"),
            schedule_sampler=create_named_schedule_sampler("uniform", model[1],  maxt=1000),
            weight_decay=experiment.get_parameter("weight_decay"),
            lr_anneal_steps=experiment.get_parameter("lr_anneal_steps"),
            save_prefix=experiment.get_parameter("save_prefix"),
            sample_step=experiment.get_parameter("sample_step"), #Sample and save an image at this step
        ).run_loop()