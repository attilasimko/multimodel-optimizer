# Place the code for brats data pre-processing here

import numpy as np
from sklearn.preprocessing import scale
from nibabel import load as load_nii
import os
import argparse
import random
import cv2
import matplotlib.pyplot as plt
import sys
import shutil

dataset = '/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/brats_mixed/'
target_dir = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0061/'

if (os.path.isdir(target_dir)):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)
os.mkdir(target_dir + "testing")
os.mkdir(target_dir + "validating")
os.mkdir(target_dir + "training")

patients = os.listdir(dataset)
patients = random.sample(patients, len(patients))
img_size = 512
pat_idx = 0
for patient in patients:
    if (pat_idx / len(patients) < 0.8):
        save_path = target_dir + 'training/'
    elif (pat_idx / len(patients) < 0.9):
        save_path = target_dir + 'validating/'
    else:
        save_path = target_dir + 'testing/'
    pat_idx = pat_idx + 1

    for dirName, subdirList, fileList in os.walk(os.path.join(dataset,patient)):
        flair = load_nii(os.path.join(dirName, patient + '_flair.nii.gz')).get_data()
        t2 = load_nii(os.path.join(dirName, patient + '_t2.nii.gz')).get_data()
        t1 = load_nii(os.path.join(dirName, patient + '_t1.nii.gz')).get_data()
        t1ce = load_nii(os.path.join(dirName, patient + '_t1ce.nii.gz')).get_data()
        labels = load_nii(os.path.join(dirName, patient + '_seg.nii.gz')).get_data()

        slc = 0
        for slc_idx in range(2, np.shape(labels)[-1] - 4, 1):
            flair_slice = np.array(cv2.resize(flair[:, :, slc_idx] / np.max(flair[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t2_slice = np.array(cv2.resize(t2[:, :, slc_idx] / np.max(t2[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1_slice = np.array(cv2.resize(t1[:, :, slc_idx] / np.max(t1[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1ce_slice = np.array(cv2.resize(t1ce[:, :, slc_idx] / np.max(t1ce[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            labels_slice = np.array(cv2.resize(labels[:, :, slc_idx], dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0, dtype=np.bool)

            np.savez(save_path + patient + '_' + str(slc),
                     flair=np.interp(flair_slice, (flair_slice.min(), flair_slice.max()), (0, 255)).astype(np.uint8),
                     t2=np.interp(t2_slice, (t2_slice.min(), t2_slice.max()), (0, 255)).astype(np.uint8),
                     t1=np.interp(t1_slice, (t1_slice.min(), t1_slice.max()), (0, 255)).astype(np.uint8),
                     t1ce=np.interp(t1ce_slice, (t1ce_slice.min(), t1ce_slice.max()), (0, 255)).astype(np.uint8),
                     mask=labels_slice)
            slc += 1