import os
from nnunet.dataset_conversion.Task001_COVID19 import convert_for_training, convert_for_testing
from nnunet.paths import base
from file_and_folder_operations import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###############################  parameters for test
fold = '0'  ## '0','1','2','3','4'
test_folder = '.../COVID-19-CT-Seg/COVID-19-CT-Seg_20cases_edit1'

###########################  Do not modify the following parameters.
task_name = '001'	## '001' for COVID-19
net_type = '3d_fullres'   ##'2d' or '3d_fullres' or '3d_lowres' or '3d_cascade_fullres'
output_folder = join(base, 'test_results', task_name, net_type, fold)
trainner = 'nnUNetTrainerV2 '
if net_type == '3d_cascade_fullres':
    trainner = 'nnUNetTrainerV2CascadeFullRes '

if __name__ == "__main__":
    print('begin conversion test...................')
    test_input_folder = convert_for_testing(test_folder, task_name)
    print('begin test...................')
    maybe_mkdir_p(output_folder)
    os.system('python ./nnunet/inference/predict_simple.py \
                -i ' + test_input_folder + ' \
                -o ' + output_folder + ' \
                -t ' + task_name + ' \
                -tr ' + trainner + ' \
                -m ' + net_type + ' \
                -f ' + fold)

