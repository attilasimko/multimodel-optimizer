import os
import sys
sys.path.extend([
    "./"
])

import pandas as pd
import numpy as np
import pydicom
import torch
import cv2
import matplotlib.pyplot as plt

from data.base_dataset import BaseDataset
from utils import util_general
from utils import util_path

def plot_img(x, pname):
    x = x.detach().cpu().numpy()
    x = x[0, :, :]
    plt.imshow(x, cmap='gray')
    plt.title(pname)
    plt.axis('off')
    plt.show()

def convert_hu_img(dicom_file):
    img = dicom_file.pixel_array
    intercept = dicom_file.RescaleIntercept
    slope = dicom_file.RescaleSlope
    img = slope * img + intercept
    return img

def clip_img(hu_img, lower, upper):
    return np.clip(hu_img, lower, upper)

def normalize_img(x, lower, upper, data_range = '-11'):
    if lower is None:
        lower = np.min(x)
    if upper is None:
        upper = np.max(x)
    x_norm = (x - lower) / (upper - lower) # map between 0 and 1

    if data_range == '01':
        return x_norm
    else:
        return (2 * x_norm) - 1

class MayoClinicDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        parser.add_argument('--modalities', help="Dataset modalities", metavar="STRING", type=str, default="full_1mm,quarter_1mm")
        parser.add_argument('--img_shape', help="Image shape for resize.", type=int, default=256)
        parser.add_argument('--plot_verbose', help="Plot images.", type=bool, default=False)
        parser.add_argument('--model_name', help="Model to use for training.", default='pix2pix')
        return parser

    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # Save the option and dataset root.
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self._path = self.opt.get_parameter("dataroot")
        self.lower = -1000
        self.upper = 2000
        self.challenge_split = 'Training_Image_Data'
        self.rec_kernel = '1mm B30'
        self.img_shape = self.opt.get_parameter("load_size")
        self.plot_verbose = self.opt.get_parameter("plot_verbose") == "True"
        self.model_name = self.opt.get_parameter("model")

        # Upload the annotations.
        df = pd.read_csv(os.path.join(self._path, f"{phase}-mayo-clinic.csv"), index_col=0)
        self.df_ld = df.loc[df['domain'] == 'LD'].reset_index(drop=True)
        self.df_hd = df.loc[df['domain'] == 'HD'].reset_index(drop=True)
        if len(self.df_ld) == 0 or len(self.df_hd) == 0:
            raise IOError("No image files found in the specified path.")
        # Define the lenght of the data.
        self.len_lw = len(self.df_ld)
        self.len_hd = len(self.df_hd)
        if self.len_lw != self.len_hd:
            raise IOError("Uncoupled dataset.")

        # Check Modalities.
        modalities = ''
        self._modalities = util_general.parse_comma_separated_list("full_1mm,quarter_1mm")
        assert len(self._modalities) > 0
        self._mode_to_idx = {mode: i for i, mode in enumerate(self._modalities)}
        self._idx_to_mode = {i: mode for mode, i in self._mode_to_idx.items()}

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """



        A_paths = self.df_hd['partial_path'].iloc[index] # High Dose
        B_paths = self.df_ld['partial_path'].iloc[index] # Low Dose
        A_idslice = util_path.get_filename(A_paths).split('.')[3]
        B_idslice = util_path.get_filename(B_paths).split('.')[3]
        assert A_idslice == B_idslice

        # Load the Dicom.
        A_dicom = pydicom.dcmread(os.path.join(self._path, self.challenge_split, self.rec_kernel, self._modalities[0], A_paths)) # high-dose
        B_dicom = pydicom.dcmread(os.path.join(self._path, self.challenge_split, self.rec_kernel, self._modalities[1], B_paths))# low-dose

        # Perform transforms.
        A_transform = self.transforms(A_dicom)
        B_transform = self.transforms(B_dicom)

        if self.plot_verbose:
            plot_img(A_transform, pname='full_1mm')
            plot_img(B_transform, pname='quarter_1mm')

        if self.model_name == 'pix2pix' or self.model_name == 'cycle_gan':
            return  {'A': A_transform, 'B': B_transform, 'A_paths': A_paths, 'B_paths': B_paths}
        elif self.model_name == 'diffusion':
            return (A_transform[None, ...], B_transform[None, ...])
        elif self.model_name == 'srresnet':
            return [A_transform[0, :, :] - B_transform[0, :, :], B_transform[0, :, :]]
        else:
            raise NotImplementedError



    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len_lw

    def transforms(self, dicom, tensor_output=True):
        x = convert_hu_img(dicom)
        x = clip_img(x, self.lower, self.upper)
        x = normalize_img(x, self.lower, self.upper)
        x = cv2.resize(x, (self.img_shape, self.img_shape))
        x = x.astype("float32")
        if tensor_output:
            x = torch.from_numpy(x)
            # x = x.unsqueeze(dim=0)
            return x
        else:
            return x