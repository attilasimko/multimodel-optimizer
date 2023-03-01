import torch
import torch.nn as nn
import math

def sct_range(x):
    import tensorflow
    from tensorflow.keras import backend as K
    x = tensorflow.where(K.greater_equal(x, -1), x, -1 * K.ones_like(x))
    x = tensorflow.where(K.less_equal(x, 1), x, 1 * K.ones_like(x))
    return x

def znorm(x):
    import tensorflow
    import tensorflow.keras.backend as K
    t_mean = K.mean(x, axis=(1, 2, 3))
    t_std = K.std(x, axis=(1, 2, 3))
    return tensorflow.math.divide_no_nan(x - t_mean[:, None, None, None], t_std[:, None, None, None])

class SRResNet():
    config = {
            "algorithm": "bayes",
            "name": "SRResNet",
            "spec": {"maxCombo": 20, "objective": "minimize", "metric": "val_loss"},
            "parameters": {
                # "first_layer_units": {
                #     "type": "integer",
                #     "mu": 500,
                #     "sigma": 50,
                #     "scalingType": "normal",
                # },
                "optimizer": {"type": "categorical", "values": ["Adam", "SGD", "RMSprop"]},
                "learning_rate": {"type": "discrete", "values": [0.001, 0.0001, 0.00001]},
                "num_filters": {"type": "integer", "min": 32, "max": 64},
                "dropout_rate": {"type": "float", "min": 0.0, "max": 0.6},
                "batch_size": {"type": "discrete", "values": [4, 8]},
            },
            "trials": 1,
        }

    def build_TF_SRResNet(experiment, task):
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout,\
        Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add , Concatenate, add, LeakyReLU
        from tensorflow.keras.models import Model
        from tensorflow.keras import activations
        from tensorflow.keras.optimizers import Adam, SGD, RMSprop
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.activations import softmax
        import math
        import numpy as np
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda, UpSampling2D, Conv2DTranspose
        from tensorflow.python.keras.layers import PReLU, ReLU

        input = Input(shape=(512, 512, 1))
        dropout_rate = experiment.get_parameter('dropout_rate')
        num_filters = experiment.get_parameter('num_filters')
            
        x = Conv2D(4, kernel_size=3, padding='same')(input)
        x = x_1 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = x_2 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = x_3 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = x_4 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = x_5 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = x_6 = LeakyReLU(0.2)(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)


        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Concatenate()([x, x_6])
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_5])
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_4])
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_3])
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_2])
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        if (dropout_rate != 0.0):
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_1])
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)

        output = Conv2D(1, kernel_size=1, padding='same')(x)
        if (task == "sct"):
            output = Activation(sct_range)(output)
        elif (task == "transfer"):
            output = Activation(znorm)(output)

        model = Model(input, output)
        if (experiment.get_parameter("optimizer") == "Adam"): # "Adam", "SGD", "RMSprop"
            model.compile(optimizer=Adam(experiment.get_parameter("learning_rate")), loss='mse', metrics=['mse'])
        elif (experiment.get_parameter("optimizer") == "SGD"):
            model.compile(optimizer=SGD(experiment.get_parameter("learning_rate")), loss='mse', metrics=['mse'])
        elif (experiment.get_parameter("optimizer") == "RMSprop"):
            model.compile(optimizer=SGD(experiment.get_parameter("learning_rate")), loss='mse', metrics=['mse'])
            
        return model

    def train(experiment, model, task, gen_train, gen_val):
        from tensorflow.keras.utils import OrderedEnqueuer
        from tensorflow.config.experimental import get_device_details, get_memory_info
        from tensorflow.config import list_physical_devices
        import numpy as np
        import utils
        
        if (utils.memory_check(experiment, model) == False):
            val_score = utils.evaluate(experiment, model, gen_val, "val", task)
            return
        tr_seq = OrderedEnqueuer(gen_train, use_multiprocessing=True)
        min_loss = np.inf
        patience = 0
        patience_thr = 5
        for epoch in range(experiment.get_parameter("epochs")):
            tr_seq.start(workers=2, max_queue_size=20)
            data_seq = tr_seq.get()
            train_loss = []
            for idx in range(100):#int(len(gen_train))):
                x_mri, x_ct = next(data_seq)
                gan_loss = model.train_on_batch(x_mri, x_ct)
                train_loss.append(gan_loss)

            gen_train.on_epoch_end()
            tr_seq.stop()
            experiment.log_metrics({"training_loss": np.mean(train_loss)}, epoch=epoch)

            val_score = utils.evaluate(experiment, model, gen_val, "val", task)
            if (val_score < min_loss):
                patience = 0
                min_loss = val_score
                print("Validation score %s", val_score)
            else:
                patience += 1
                if patience > patience_thr:
                    print("Early stopping")
                    break

class PT_SRResNet:
    # https://github.com/twtygqyy/pytorch-SRResNet
    class _Residual_Block(nn.Module):
        def __init__(self):
            super(PT_SRResNet._Residual_Block, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            self.in1 = nn.InstanceNorm2d(64, affine=True)
            self.relu = nn.LeakyReLU(0.2, inplace=True)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            self.in2 = nn.InstanceNorm2d(64, affine=True)

        def forward(self, x):
            identity_data = x
            output = self.relu(self.in1(self.conv1(x)))
            output = self.in2(self.conv2(output))
            output = torch.add(output,identity_data)
            return output 
    class _SRResNet(nn.Module):
        def __init__(self, experiment):
            super(PT_SRResNet._SRResNet, self).__init__()

            self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
            self.relu = nn.LeakyReLU(0.2, inplace=True)
            
            self.residual = self.make_layer(PT_SRResNet._Residual_Block, 16)

            self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_mid = nn.InstanceNorm2d(64, affine=True)

            self.upscale4x = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()

        def make_layer(self, block, num_of_layer):
            layers = []
            for _ in range(num_of_layer):
                layers.append(block())
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.relu(self.conv_input(x))
            residual = out
            out = self.residual(out)
            out = self.bn_mid(self.conv_mid(out))
            out = torch.add(out,residual)
            out = self.upscale4x(out)
            out = self.conv_output(out)
            return out

    class _NetD(nn.Module):
        def __init__(self):
            super(PT_SRResNet._NetD, self).__init__()

            self.features = nn.Sequential(
            
                # input is (3) x 96 x 96
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (64) x 96 x 96
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),            
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (64) x 96 x 96
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),            
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                
                # state size. (64) x 48 x 48
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (128) x 48 x 48
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (256) x 24 x 24
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (256) x 12 x 12
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),            
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (512) x 12 x 12
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),            
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
            self.fc1 = nn.Linear(512 * 6 * 6, 1024)
            self.fc2 = nn.Linear(1024, 1)
            self.sigmoid = nn.Sigmoid()

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)

        def forward(self, input):

            out = self.features(input)

            # state size. (512) x 6 x 6
            out = out.view(out.size(0), -1)

            # state size. (512 x 6 x 6)
            out = self.fc1(out)

            # state size. (1024)
            out = self.LeakyReLU(out)

            out = self.fc2(out)
            out = self.sigmoid(out)
            return out.view(-1, 1).squeeze(1)