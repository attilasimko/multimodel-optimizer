import os
import sys
sys.path.extend([
    "./"
])
import zipfile

raw_dir = "/data_m2/lorenzo/data/raw"
dataset_name = 'mayo-clinic'
challenge_split = 'Training_Image_Data'
rec_kernel = '1mm B30'
modalities = ['full_1mm', 'quarter_1mm']

path_zipfile = os.path.join(raw_dir, dataset_name, challenge_split, rec_kernel)
with zipfile.ZipFile(os.path.join(path_zipfile, 'FD_1mm.zip'), "r") as z:
  z.extractall(path_zipfile)
print('FD Done.')

with zipfile.ZipFile(os.path.join(path_zipfile, 'QD_1mm.zip'), "r") as z:
  z.extractall(path_zipfile)
print('QD Done.')