import os
import pickle5 as pickle
import zipfile

import numpy as np

from data.base_dataset import BaseDataset
from utils import util_general


class PelvisDataset(BaseDataset):
    """A dataset class for paired medical image dataset.

        It assumes that the directory '/path/to/data/train' contains image pairs in the form of [{A,B}, H, W].
        During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        parser.add_argument('--modalities', help="Dataset modalities", metavar="STRING", type=str, default="MR_nonrigid_CT,MR_MR_T2")
        return parser

    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # Save the option and dataset root.
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self._path = opt.get_parameter("dataroot")
        self._modalities = ['MR_MR_T2', 'MR_nonrigid_CT']
        
        assert len(self._modalities) > 0
        self._mode_to_idx = {mode: i for i, mode in enumerate(self._modalities)}
        self._idx_to_mode = {i: mode for mode, i in self._mode_to_idx.items()}

        # Check zipfile.
        self._zipfile = None
        if self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        # Get the image paths.
        self.AB_paths = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == ".pickle" and  phase in fname)
        if len(self.AB_paths) == 0:
            raise IOError("No image files found in the specified path")

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
        # read a image given a random integer index

        # Load Multimodal images dictionary.
        AB_path = self.AB_paths[index]
        with self._open_file(AB_path) as f:
            AB_dict = pickle.load(f)
        AB = self._create_AB(AB_dict)

        # Sanity checks.
        assert AB.dtype == np.dtype('float32')
        assert isinstance(AB, np.ndarray)
        assert AB.shape == (len(self._modalities), self.opt.get_parameter("load_size"), self.opt.get_parameter("load_size"))

        # Select image A and B.
        A = AB[self._mode_to_idx['MR_nonrigid_CT'], :, :].astype("float32")  # CT
        B = AB[self._mode_to_idx['MR_MR_T2'], :, :].astype("float32")  # MRI

        # Perform transforms.
        A_transform = self.transform_ct(A)
        B_transform = self.transform_mr(B)


        model = self.opt.get_parameter("model")
        if model == "srresnet":
            return [A_transform, B_transform]
        elif model == "pix2pix":
            return {'A': A_transform, 'B': B_transform, 'A_paths': AB_path, 'B_paths': AB_path}
        elif model == "diffusion":
            return (A_transform[None, ...], B_transform[None, ...])

    def transform_mr(self, img):
        img = img - np.mean(img)
        img = img / np.std(img)
        return img
    
    def transform_ct(self, img):
        img = (img / 85) - 1.0
        return img
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    # Added functions to manage zip file
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        else:
            raise IOError("Support only zip.")

    def _create_AB(self, p):
        s = p[self._modalities[0]]
        out_image = np.zeros((len(self._modalities), s.shape[0], s.shape[1])).astype("float32") # Compose the Multichannel image.
        for i, _modality in enumerate(self._modalities):
            x = p[_modality]
            x = x.astype("float32")
            out_image[i, :, :] = x
        return out_image

    def _load_img(self, index):
        AB_path = self.AB_paths[index]
        with self._open_file(AB_path) as f:
            AB_dict = pickle.load(f)
        AB = self._create_AB(AB_dict)

        # Sanity checks.
        assert AB.dtype == np.dtype('float32')
        assert isinstance(AB, np.ndarray)
        assert AB.shape == (len(self._modalities), self.opt.get_parameter("load_size"), self.opt.get_parameter("load_size"))

        # Select image A and B.
        A = AB[self._mode_to_idx['MR_MR_T2'], :, :].astype("float32")  # MRI
        B = AB[self._mode_to_idx['MR_nonrigid_CT'], :, :].astype("float32")  # CT

        return A, B, AB_path