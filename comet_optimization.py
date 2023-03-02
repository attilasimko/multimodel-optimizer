import comet_ml
import models
import utils
import argparse
import os
comet_ml.init(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name='comet-optimizer')

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--gpu", default=None)
parser.add_argument("--task", default="sct") # sct / denoise / transfer / time
parser.add_argument("--model", default="srresnet") # srresnet / pix2pix / diffusion
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if (args.model == "srresnet"):
    config = models.SRResNet.config
opt = comet_ml.Optimizer(config)

experiment_idx = 0
for experiment in opt.get_experiments():
    experiment.set_name(f"{args.task}_{args.model}_{experiment_idx}")
    experiment_idx += 1
    experiment.log_parameter("task", args.task)
    experiment.log_parameter("model", args.model)
    experiment.log_parameter("epochs", 10)
    experiment.log_parameter("workers", 4)
    experiment.log_parameter("max_queue_size", 4)
    experiment.log_parameter("use_multiprocessing", "False")
    gen_train, gen_val, gen_test = utils.setup_generators(experiment, args.task)

    # Build the model:
    if (args.model == "srresnet"):
        # model = models._SRResNet(experiment)
        model = models.SRResNet.build_TF_SRResNet(experiment, args.task, experiment.get_parameter('dropout_rate'))
    else:
        raise Exception("Unknown model")

    # Train the model:
    if (args.model == "srresnet"):
        models.SRResNet.train(experiment, model, args.task, gen_train, gen_val)
    else:
        raise Exception("Unknown model")

    # How well did it do?
    utils.plot_results(experiment, model, gen_val)
    utils.evaluate(experiment, model, gen_test, "test", args.task)
    experiment.end()