from .base_options import BaseOptions
from .comet_options import CometOptions

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        self.parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        self.parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.parser = CometOptions.initialize(self,self.parser)

        self.isTrain = True
        return self.parser
