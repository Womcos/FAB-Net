import shutil
from collections import OrderedDict
import numpy as np
import SimpleITK as sitk
import multiprocessing
from nnunet.paths import nnUNet_raw_data
from file_and_folder_operations import *

## The directions of data and label are different. When we use
## SimpleITK to get the ndarray, we should add the direction information
## to change the image ndarray.

def convert_for_training(dirs, task_name):
    # dirs = ['F:/NIH_data/pancreas-NIH/Pancreas-CT/data',
    #        'F:/NIH_data/pancreas-NIH/Pancreas-CT/TCIA_pancreas_labels-02-05-2017']
    task_id = 1
    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_base = join(nnUNet_raw_data, foldername)

    train_folder = join(out_base, 'imagesTr')
    label_folder = join(out_base, 'labelsTr')
    test_folder = join(out_base, 'imagesTs')

    maybe_mkdir_p(train_folder)
    maybe_mkdir_p(label_folder)
    maybe_mkdir_p(test_folder)

    img_data = os.listdir(dirs[0])
    lab_data = os.listdir(dirs[1])
    assert len(img_data) == len(lab_data)

    for img in img_data:
        ori_img = join(dirs[0], img)
        tar_img = join(train_folder, img.replace('.nii.gz', '_0000.nii.gz'))
        shutil.copy(ori_img, tar_img)

    for lab in lab_data:
        ori_lab = join(dirs[1], lab)
        tar_lab = join(label_folder, lab)
        shutil.copy(ori_lab, tar_lab)

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see COVID19 website"
    json_dict['licence'] = "see COVID19 website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "OTHER"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "infections",
    }
    json_dict['numTraining'] = len(os.listdir(label_folder))
    json_dict['numTest'] = len(os.listdir(test_folder))

    json_dict['training'] = []
    json_dict['test'] = []

    train_img = os.listdir(train_folder)
    train_lab = os.listdir(label_folder)
    assert len(train_img) == len(train_lab)
    test_img = os.listdir(test_folder)

    for lab in train_lab:
        tar_lab = join(label_folder, lab)
        tar_img = join(train_folder, lab)
        json_dict['training'].append({'image':tar_img, 'label':tar_lab})

    for img in test_img:
        tar_img = join(test_folder, img)
        json_dict['test'].append(tar_img)

    save_json(json_dict, join(out_base, "dataset.json"))

def convert_for_testing(test_dir, task_name):
    task_id = 1
    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_base = join(nnUNet_raw_data, foldername)

    test_folder = join(out_base, 'imagesTs')
    maybe_mkdir_p(test_folder)

    img_data = os.listdir(test_dir)
    for img in img_data:
        ori_img = join(test_dir, img)
        tar_img = join(test_folder, img.replace('.nii.gz', '_0000.nii.gz'))
        shutil.copy(ori_img, tar_img)
    return test_folder
