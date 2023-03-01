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

def memory_check(experiment, model):
    import nvidia_smi
    import numpy as np

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    available_memory = np.round(info.total / (1024.0 ** 3), 3)
    nvidia_smi.nvmlShutdown()

    required_memory = get_TF_memory_usage(experiment.get_parameter("batch_size"), model) 
    experiment.log_parameter("reqmemory", required_memory)

    if (required_memory > available_memory):
        print(f"ERROR: Not enough memory. Required: {required_memory} GB, Available: {available_memory} GB")
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
        pred = model.predict_on_batch(x_mri)
        loss = 1000 * np.abs(pred - x_ct[0])[x_ct[0]>-1]
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