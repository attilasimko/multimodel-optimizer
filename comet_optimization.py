import comet_ml
import models
import utils
import argparse

comet_ml.init(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name='comet-optimizer')

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--task", default="sct") # sct / denoise / segment
parser.add_argument("--model", default="srresnet") # srresnet / pix2pix / diffusion
args = parser.parse_args()

if (args.model == "srresnet"):
    config = models.SRResNet.config
opt = comet_ml.Optimizer(config)

if (args.task == "sct"):
    data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0060/'
else:
    raise Exception("Unknown task")

experiment_idx = 0
for experiment in opt.get_experiments():
    experiment.set_name(f"{args.task}_{args.model}_{experiment_idx}")
    experiment_idx += 1
    experiment.log_parameter("task", args.task)
    experiment.log_parameter("model", args.model)
    gen_train, gen_val, gen_test = utils.setup_generators(experiment, data_path)
    experiment.log_parameter("epochs", 10)

    # Build the model:
    if (args.model == "srresnet"):
        # model = models._SRResNet(experiment)
        model = models.SRResNet.build_TF_SRResNet(experiment)
    else:
        raise Exception("Unknown model")

    models.SRResNet.train(experiment, model, gen_train, gen_val)

    # How well did it do?
    utils.plot_results(experiment, model, gen_val)
    utils.evaluate(experiment, model, gen_test, "test")
    experiment.end()