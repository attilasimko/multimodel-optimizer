import comet_ml
comet_ml.init(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name='comet-optimizer', workspace="attilasimko")
import utils_misc
import argparse
from data import create_dataset
from models import create_model
from models import SRResNet
from models import pix2pix_model
from models import cycle_gan_model
import os

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--gpu", default=None)
parser.add_argument("--log_comet", default="False")
parser.add_argument("--task", default="denoise") # sct / denoise / transfer
parser.add_argument("--model", default="srresnet") # srresnet / pix2pix / cycle_gan / diffusion
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if args.model == "srresnet":
    config = SRResNet.config
elif args.model == "pix2pix":
    config = pix2pix_model.config
elif args.model == "cycle_gan":
    raise NotImplementedError
elif args.model == "diffusion":
    raise NotImplementedError # todo parameters diff salih
else:
    raise Exception("Unknown model")
log_comet = args.log_comet == "False"

opt = comet_ml.Optimizer(config)

experiment_idx = 0
for experiment in opt.get_experiments(disabled=log_comet):
    dataroot = utils_misc.get_dataset_path(experiment, args.task)
    experiment.set_name(f"{args.task}_{args.model}_{experiment_idx}")
    experiment_idx += 1
    experiment.log_parameter("task", args.task)
    experiment.log_parameter("model", args.model)
    experiment.log_parameter("load_size", 256)
    experiment.log_parameter("dataroot", dataroot)
    experiment.log_parameter("epochs", 100)
    experiment.log_parameter("max_dataset_size", 1000000)
    experiment.log_parameter("workers", 4)
    experiment.log_parameter("max_queue_size", 4)
    experiment.log_parameter("use_multiprocessing", "False")
    experiment.log_parameter("plot_verbose", "False")
    gen_train, gen_val, gen_test = create_dataset(experiment)

    # Build the model:
    if args.model == "srresnet":
        model = SRResNet.build_TF_SRResNet(experiment, args.task, experiment.get_parameter('dropout_rate'))
    elif args.model == "pix2pix":
        model = create_model(experiment)  # create a model given opt.model and other options
        model.setup(opt)
    elif args.model == "cycle_gan":
        raise NotImplementedError # todo build cyclegan lorenzo
    elif args.model == "diffusion":  # todo build diff salih
        raise NotImplementedError
    else:
        raise Exception("Unknown model")

    # Train the model:
    if args.model == "srresnet":
        SRResNet.train(experiment, model, args.task, gen_train, gen_val)
    elif args.model == "pix2pix":
        raise NotImplementedError # todo pix2pix training lorenzo
    elif args.model == "cycle_gan":
        raise NotImplementedError # todo cyclegan training lorenzo
    elif args.model == "diffusion":
        raise NotImplementedError # todo diff training salih
    else:
        raise Exception("Unknown model")

    # How well did it do?
    utils_misc.plot_results(experiment, model, gen_val)
    utils_misc.evaluate(experiment, model, gen_test, "test", args.task)
    experiment.end()