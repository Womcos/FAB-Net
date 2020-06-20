import SimpleITK as sitk
import numpy as np
import os


Path = 'F:/COVID-19-CT-Seg/COVID-19-CT-Seg_20cases'
if not os.path.exists('F:/COVID-19-CT-Seg/COVID-19-CT-Seg_20cases_edit1'):
    os.makedirs('F:/COVID-19-CT-Seg/COVID-19-CT-Seg_20cases_edit1')

data_name = 'coronacases'
name_list = []
for i in range(10):
    if i == 0:
        name_list.append(Path + '/' + data_name + '_010.nii.gz')
    else:
        name_list.append(Path + '/' + data_name + '_00' + str(i) + '.nii.gz')

print(name_list)
for name in name_list:
    print(name)
    output_name = name.replace('COVID-19-CT-Seg_20cases', 'COVID-19-CT-Seg_20cases_edit1')
    img = sitk.ReadImage(name)
    img_arr = sitk.GetArrayFromImage(img)
    tar_arr = np.array(img_arr, dtype=np.float32)
    tar_arr[tar_arr > 250] = 250
    tar_arr[tar_arr < -1250] = -1250
    tar_arr = tar_arr + 1250
    tar_arr = tar_arr / 1500 * 255
    tar_arr = np.array(tar_arr, dtype=np.uint8)
    tar = sitk.GetImageFromArray(tar_arr)
    tar.CopyInformation(img)
    sitk.WriteImage(tar, output_name)
