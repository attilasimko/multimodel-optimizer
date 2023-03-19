import comet_ml
comet_ml.init(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name='comet-optimizer', workspace="attilasimko")
import utils_misc
import argparse

import os
import time
import numpy as np

from data import create_dataset
from models import create_model
from models import srresnet_model
from models import pix2pix_model
from models import cycle_gan_model

from options.train_options import TrainOptions

opt = TrainOptions().parse()  # get options

if opt.gpu_ids is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)

if opt.model == "srresnet":
    config = srresnet_model.config
elif opt.model == "pix2pix":
    config = pix2pix_model.config
elif opt.model == "cycle_gan":
    raise NotImplementedError
elif opt.model == "diffusion":
    raise NotImplementedError
else:
    raise Exception("Unknown model")
log_comet = opt.log_comet == "False"

opt_comet = comet_ml.Optimizer(config)

experiment_idx = 0
for experiment in opt_comet.get_experiments(disabled=log_comet):
    dataroot = utils_misc.get_dataset_path(experiment, opt.task)
    experiment.set_name(f"{opt.task}_{opt.model}_{experiment_idx}")
    experiment_idx += 1
    experiment.log_parameter("task", opt.task)
    experiment.log_parameter("model", opt.model)
    experiment.log_parameter("dataroot", dataroot)
    experiment.log_parameter("load_size", opt.load_size)
    experiment.log_parameter("max_dataset_size", 1000000)
    experiment.log_parameter("workers", 4)
    experiment.log_parameter("max_queue_size", 4)
    experiment.log_parameter("use_multiprocessing", "False")
    experiment.log_parameter("plot_verbose", "False")
    gen_train, gen_val, gen_test = create_dataset(experiment)

    # Build the model
    if opt.model == "srresnet":
        model = srresnet_model.build_TF_SRResNet(experiment, opt.task, experiment.get_parameter('dropout_rate'))
    elif opt.model == "pix2pix":
        opt.lr = experiment.get_parameter('lr')
        opt.n_epochs = experiment.get_parameter('n_epochs')
        opt.n_epochs_decay = experiment.get_parameter('n_epochs_decay')
        opt.gan_mode = experiment.get_parameter('gan_mode')
        opt.batch_size = experiment.get_parameter('batch_size')

        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)
    elif opt.model == "cycle_gan":
        raise NotImplementedError
    elif opt.model == "diffusion":
        raise NotImplementedError
    else:
        raise Exception("Unknown model")

    # Train the model:
    if opt.model == "srresnet":
        srresnet_model.train(experiment, model, opt.task, gen_train, gen_val)
    elif opt.model == "pix2pix":
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            model.update_learning_rate()  # update learning rates in the beginning of every epoch.

            tic = time.perf_counter()
            train_l1loss = []
            for i, data in enumerate(gen_train):
                # inner loop within one epoch

                #if i > 5:
                #    break

                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
                losses = model.get_current_losses()
                train_l1loss.append(losses['G_L1'])

            toc = time.perf_counter()
            experiment.log_metrics({"training_loss": np.mean(train_l1loss), "epoch_time [s]": toc - tic}, epoch=epoch)

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d' % epoch)
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, toc - tic))
    elif opt.model == "cycle_gan":
        raise NotImplementedError
    elif opt.model == "diffusion":
        raise NotImplementedError
    else:
        raise Exception("Unknown model")

    # How well did it do?
    utils_misc.plot_results(experiment, model, gen_val)
    utils_misc.evaluate(experiment, model, gen_test, "test", opt.task)
    experiment.end()