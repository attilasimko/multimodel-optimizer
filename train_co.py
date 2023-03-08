import sys
sys.path.extend(["./"])

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

from utils import util_general
import pickle
import os

import matplotlib.pyplot as plt

if __name__ == '__main__':

    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    history = util_general.list_dict()
    history_epoch = util_general.list_dict()
    report_dir = os.path.join(opt.checkpoints_dir, opt.name)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        history_runtime =  util_general.list_dict()
        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            # For each batch we update the history.
            for k in list(losses.keys()):
                history[k].append(losses[k])
                history_runtime[k].append(losses[k] * data['A'].shape[0])

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)
                print('losses snap at the end of epoch %d, iters %d' % (epoch, total_iters))
                with open(os.path.join(report_dir, 'history'), 'wb') as handle:
                    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(report_dir, 'history_epoch'), 'wb') as handle:
                    pickle.dump(history_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print('End of epoch %d / %d \t Time Taken: %d sec' % (
                epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))