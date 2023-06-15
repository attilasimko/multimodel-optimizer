# Copyright © Lorenzo Tronchin
#
# V1
# Last Update 31-10-2022
# Contact lorenzo.tronchin@umu.se
#

import pickle
import os
import zipfile

import numpy as np
import torch

def normalize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, path, resolution=256, split_flag="train"):

        self._path = os.path.join(path, 'Pelvis_2.1_repo_no_mask', 'Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10' + '.zip')
        self._modalities =  parse_comma_separated_list('MR_MR_T2,MR_nonrigid_CT')
        self._zipfile = None
        self.split_flag=split_flag

      
        # Check if the dataset is stored as zip file.
        if self._file_ext(self._path) == ".zip":
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a zip")

        # Extract all the filenames stored in the zip file.
        self._fnames = [
            fname for fname in self._all_fnames if self._file_ext(fname) == ".pickle" and self.split_flag in fname
        ]
        self._fnames = sorted(self._fnames)
        if len(self._fnames) == 0:
            raise IOError("No files found in the specified path")

        # Sanity checks.
        raw_shape = [len(self._fnames)] + list(self._load_raw_image(raw_idx=0)[0].shape)
        #print(len(self._fnames))
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        if len(self._modalities) is not None and (raw_shape[1] != len(self._modalities)):
            raise IOError("Image does not match the specified number of channels.")

        # Define the list of possible indexes.
        self._raw_shape = list(raw_shape)
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        img, fname = self._load_raw_image(self._raw_idx[idx])
        if self.split_flag == "test":
            img = normalize(img)
            return (img[:-1, ...], fname)
        elif self.split_flag == "train":
            w = img[:-1, ...]
            label = img[-1, ...][None, ...]
            w = normalize(w)
            label = normalize(label)
            return (w, label)
        elif self.split_flag == "val":
            w = img[:-1, ...]
            label = img[-1, ...][None, ...]
            w = normalize(w)
            label = normalize(label)
            return (w, label)
        else:
            raise IOError("Choose a split: 'train', 'val', 'test'." )

    # Zip utilities.
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile  

    def _open_file(self, fname):
        return self._get_zipfile().open(fname, "r")

    def _load_raw_image(self, raw_idx):
        fname = self._fnames[raw_idx]

        with self._open_file(fname) as f:
            p = pickle.load(f)

        assert len(self._modalities) > 0
        s = p[self._modalities[0]]

        out_image = np.zeros((len(self._modalities), s.shape[0], s.shape[1])).astype("float32") # compose the multichannel image 2x256x256
        for i, _modality in enumerate(self._modalities):
            x = p[_modality]
            x = x.astype("float32")
            out_image[i, :, :] = x

        return out_image, fname # CHW
        
if __name__ == '__main__':

    interim_dir = '/Users/ltronchin/Desktop/Cosbi/Computer Vision/Gan-track/data/interim/' #'/home/lorenzo/data/interim/' # INSERT THE PATH HERE
    dataset = 'Pelvis_2.1_repo_no_mask'
    dataset_name = 'Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10'
    modalities =  parse_comma_separated_list('MR_MR_T2,MR_nonrigid_CT') #'MR_nonrigid_CT,MR_MR_T2'
    phase = 'train'
    res = 256
    batch_size=16

    img_dataset = ImgDataset(
        path=os.path.join(interim_dir, dataset, dataset_name + '.zip'), modalities=modalities, split=phase, resolution=res
    )

    data_loader = torch.utils.data.DataLoader(dataset=img_dataset, batch_size=batch_size, shuffle=True)

    for x, fname in data_loader:
        print(x.shape)
        print(fname)
        # ... your training procedure