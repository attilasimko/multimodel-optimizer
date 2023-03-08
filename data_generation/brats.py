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

def znorm(img):
    img = img - np.mean(img)
    img = img / np.std(img)
    return img

dataset = '/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/BRATS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/'
target_dir = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0061/'

if (os.path.isdir(target_dir)):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)
os.mkdir(target_dir + "testing")
os.mkdir(target_dir + "validating")
os.mkdir(target_dir + "training")

patients = os.listdir(dataset)
patients = random.sample(patients, len(patients))
img_size = 256
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
            flair_slice = znorm(np.array(cv2.resize(flair[:, :, slc_idx] / np.max(flair[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single))
            t2_slice = znorm(np.array(cv2.resize(t2[:, :, slc_idx] / np.max(t2[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single))
            t1_slice = znorm(np.array(cv2.resize(t1[:, :, slc_idx] / np.max(t1[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single))
            t1ce_slice = znorm(np.array(cv2.resize(t1ce[:, :, slc_idx] / np.max(t1ce[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single))
            labels_slice = np.array(cv2.resize(labels[:, :, slc_idx], dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0, dtype=np.bool)

            if (np.isnan(flair_slice * t2_slice * t1_slice * t1ce_slice).any()):
                continue

            if (np.sum(t1_slice > 0.05 * (np.max(t1_slice) - np.min(t1_slice))) < img_size*img_size*0.25):
                continue

            np.savez(save_path + patient + '_' + str(slc),
                     flair=flair_slice,
                     t2=t2_slice,
                     t1=t1_slice,
                     t1ce=t1ce_slice,
                     mask=labels_slice)
            slc += 1