def setup_generators(experiment, task):
    import data
    
    data_path = get_dataset_path(experiment, task)
    if task == "sct":
        input = "mr"
        output = "ct"
        gen_train = data.DataGenerator(data_path + "training/",
                            inputs=[[input, False, 'float32']],
                            outputs=[[output, False, 'float32']],
                            batch_size=experiment.get_parameter("batch_size"),
                            shuffle=True)
        gen_val = data.DataGenerator(data_path + "validating/",
                                inputs=[[input, False, 'float32']],
                                outputs=[[output, False, 'float32']],
                                batch_size=experiment.get_parameter("batch_size"),
                                shuffle=False)
        gen_test = data.DataGenerator(data_path + "testing/",
                                inputs=[[input, False, 'float32']],
                                outputs=[[output, False, 'float32']],
                                batch_size=experiment.get_parameter("batch_size"),
                                shuffle=False)
    
    
    return gen_train, gen_val, gen_test

def memory_check(experiment, model):
    import nvidia_smi
    import numpy as np

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    available_memory = np.round(info.total / (1024.0 ** 3), 3) # We could just hard-set this to 10GB. That's the limit on the smallest GPUs we have.
    nvidia_smi.nvmlShutdown()

    required_memory = get_TF_memory_usage(experiment.get_parameter("batch_size"), model) 
    experiment.log_parameter("reqmemory", required_memory)

    if required_memory > available_memory:
        print(f"ERROR: Not enough memory. Required: {required_memory} GB, Limit: {available_memory} GB")
        return False
    return True

def get_TF_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_TF_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def get_dataset_path(experiment, task):
    import os
    if os.path.isdir('/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/data/'): # If running on my local machine
        data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/data/'
        experiment.log_parameter("server", "GERTY")
    elif os.path.isdir('/data_m2/lorenzo/data/'): # If running on laplace
        data_path = '/data_m2/lorenzo/data/'
        experiment.log_parameter("server", "laplace")
    elif os.path.isdir('/data/attila/data/'): # If running on gauss
        data_path = '/data/attila/data/'
        experiment.log_parameter("server", "gauss")
    else:
        raise Exception("Unknown server")

    if task == "sct":
        data_path += 'interim/Pelvis_2.1_repo_no_mask/Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10.zip'
    elif task == "transfer":
        data_path += 'interim/brats/brats.zip'
    elif task== "denoise":
        data_path += 'interim/mayo-clinic/'
    else:
        raise Exception("Unknown task")
    
    return data_path
    
def evaluate(experiment, model, gen, eval_type, task):
    import numpy as np
    from tensorflow.keras.utils import OrderedEnqueuer
    
    loss_list = []
    for i, data in enumerate(gen):
        if task == "sct":
            x_ct = np.expand_dims(data[0].numpy(), 3)
            x_mri = np.expand_dims(data[1].numpy(), 3)
            pred = model.predict_on_batch(x_mri)
            loss = 1000 * np.abs(pred - x_ct)[x_ct>-1]
            loss_list.extend(np.mean(loss))
        elif task == "transfer":
            x_t1ce = np.expand_dims(data[0].numpy(), 3)
            x_t1 = np.expand_dims(data[1].numpy(), 3)
            loss = model.test_on_batch(x_t1, x_t1ce)
            loss_list.extend(loss)
        elif task == "denoise":
            x_hr = np.expand_dims(data[0].numpy(), 3)
            x_lr = np.expand_dims(data[1].numpy(), 3)
            pred = model.predict_on_batch(x_lr)
            loss = 1000 * np.abs(pred - x_hr)
            loss_list.extend(np.mean(loss))

    experiment.log_metrics({eval_type + "_loss": np.mean(loss_list)})
    return np.mean(loss_list)

def plot_results(experiment, model, gen):
    import numpy as np
    import matplotlib.pyplot as plt
    plot_idx = 0
    plot_num = 10
    for i, data in enumerate(gen):
        if (plot_idx <= plot_num):
            plot_idx += 1
            y = np.expand_dims(data[0].numpy(), 3)
            x = np.expand_dims(data[1].numpy(), 3)
            pred = model.predict_on_batch(x)
            if experiment.get_parameter("task") == "denoise":
                y = y - x
                pred = pred - x
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(x[0, :, :, 0], cmap='gray')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(132)
            plt.imshow(y[0, :, :, 0], cmap='gray')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(133)
            plt.imshow(pred[0, :, :, 0], cmap='gray')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            experiment.log_figure(figure=plt, figure_name="results_" + str(i), overwrite=True)
            plt.close('all')
    
def export_weights_to_hero(model):
    return