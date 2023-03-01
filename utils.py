def setup_generators(experiment, task):
    import data
    data_path = get_dataset_path(experiment, task)

    gen_train = data.DataGenerator(data_path + "training/",
                            inputs=[['mr', False, 'float32']],
                            outputs=[['ct', False, 'float32']],
                            batch_size=experiment.get_parameter("batch_size"),
                            shuffle=True)
    gen_val = data.DataGenerator(data_path + "validating/",
                            inputs=[['mr', False, 'float32']],
                            outputs=[['ct', False, 'float32']],
                            batch_size=experiment.get_parameter("batch_size"),
                            shuffle=False)
    gen_test = data.DataGenerator(data_path + "testing/",
                            inputs=[['mr', False, 'float32']],
                            outputs=[['ct', False, 'float32']],
                            batch_size=experiment.get_parameter("batch_size"),
                            shuffle=False)
    
    return gen_train, gen_val, gen_test

def get_dataset_path(experiment, task):
    import os
    if (os.path.isdir('/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/')): # If running on my local machine
        data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/'
        if (task == "sct"):
            data_path += 'DS0060/'
        else:
            raise Exception("Unknown task")
        experiment.log_parameter("server", "GERTY")
    elif (os.path.isdir('/data/attila/')): # If running on laplace / gauss / neumann
        data_path = '/data/attila/'
        if (task == "sct"):
            data_path += 'DS0060/'
        else:
            raise Exception("Unknown task")
        experiment.log_parameter("server", "cluster")
    return data_path
    
def evaluate(experiment, model, gen, eval_type):
    import numpy as np
    from tensorflow.keras.utils import OrderedEnqueuer
    
    test_seq = OrderedEnqueuer(gen, use_multiprocessing=False)
    test_seq.start(workers=4, max_queue_size=10)
    data_seq = test_seq.get()
    loss_list = []
    for idx in range(int(len(gen))):
        x_mri, x_ct = next(data_seq)
        loss = model.test_on_batch(x_mri, x_ct)
        loss_list.append(loss)

    experiment.log_metrics({eval_type + "_loss": np.mean(loss_list)})
    gen.on_epoch_end()
    test_seq.stop()
    return np.mean(loss_list)

def plot_results(experiment, model, gen):
    import matplotlib.pyplot as plt
    mr, ct = gen[0]
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(mr[0][0, :, :, 0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(132)
    plt.imshow(ct[0][0, :, :, 0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(model.predict_on_batch(mr)[0, :, :, 0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    experiment.log_figure(figure=plt, figure_name="results", overwrite=True)
    plt.close('all')
    
def export_weights_to_hero(model):
    return