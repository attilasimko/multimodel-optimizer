# Place the code for creating sct dataset here

import shutil
from distutils.dir_util import copy_tree
import os
from distutils.log import fatal
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pydicom
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import cv2
from tensorflow.keras.models import load_model
import sys
sys.path.append("../")
from tensorflow import function, TensorSpec
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import concatenate
from tensorflow import io
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
np.random.seed(2001)

base_dir = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/Pelvis_2.1/"
temp_dir = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/Pelvis_2.2/"
target_dir = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0060/'

# for group in os.listdir(base_dir):
#     group_dir = base_dir + group + "/"
#     for patient in os.listdir(group_dir):
#         patient_dir = group_dir + patient + "/"
#         for scandate in os.listdir(patient_dir):
#             scandate_dir = patient_dir + scandate + "/"
#             for scan in os.listdir(scandate_dir):
#                 if (scan == "MR_MR_T2_BC"):
#                     copy_tree(scandate_dir + scan, temp_dir + patient + "/MR")
#                 if (scan == "MR_nonrigid_CT"):
#                     copy_tree(scandate_dir + scan, temp_dir + patient + "/CT")



def make_mask(img, thr):
    mask = img >= thr
    mask = ndimage.binary_dilation(mask, iterations=2)
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask = sizes > 512*512*0.08
    mask = mask[label_im]
    mask = ndimage.binary_fill_holes(mask)
    return mask

def crop_image(img, mask, defval):
    img[~mask] = defval
    return img

if (os.path.isdir(target_dir)):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)
os.mkdir(target_dir + "testing")
os.mkdir(target_dir + "validating")
os.mkdir(target_dir + "training")

patient_idx = 0
lst = os.listdir(temp_dir)
lst.sort()
np.random.shuffle(lst)

for patient in lst:
    save_path = ""
    mult = 1
    MR_STACK = []
    CT_STACK = []

    if (patient_idx / len(lst) < 0.8):
        save_path =  target_dir + 'training/'
    elif (patient_idx / len(lst) < 0.9):
        save_path = target_dir + 'validating/'
    else:
        save_path = target_dir + 'testing/'

    if (not(os.path.isdir(os.path.join(temp_dir, patient)))):
        print("Patient does not exist (" + str(patient) + ")")
        continue

    for contrast in os.listdir(os.path.join(temp_dir, patient)):
        if ((contrast == "MR") | (contrast == "CT")):
            STACK = []
            for scan_file in os.listdir(os.path.join(temp_dir, patient, contrast)):
                data = pydicom.dcmread(os.path.join(temp_dir, patient, contrast, scan_file))
                STACK.append(data)
            
            if (contrast == "MR"):
                MR_STACK = STACK
            elif (contrast == "CT"):
                CT_STACK = STACK

            if ((len(MR_STACK) > 0) & (len(CT_STACK) > 0)):
                if ((len(MR_STACK) == len(CT_STACK))):
                    CT_STACK = sorted(CT_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                    MR_STACK = sorted(MR_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                    
                    print(str(patient) + "\t" + str(len(CT_STACK)))
                    for i in range(len(MR_STACK)):
                        ct = (CT_STACK[i].RescaleIntercept + CT_STACK[i].RescaleSlope * CT_STACK[i].pixel_array) / 1000
                        ct = np.clip(cv2.resize(ct, (512, 512)), -1, 1)
                        mask = make_mask(ct, -0.2)
                        ct = crop_image(ct, mask, -1)

                        mr =  MR_STACK[i].pixel_array / np.mean(MR_STACK[i].pixel_array)
                        mr = cv2.resize(mr, (512, 512))
                        mr = crop_image(mr, mask, 0)
                        if ((np.max(mr) == 0) | (np.isnan(mr).any())): 
                            continue
                        mr =  (mr - np.mean(mr)) / np.std(mr)


                        np.savez(save_path + str.join("_", (patient, str(i))),
                                mr=np.array(mr, dtype=np.float32),
                                ct=np.array(ct, dtype=np.float32))
    patient_idx += 1
only_training = True