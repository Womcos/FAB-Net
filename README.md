# FAB-Net

FAB-Net represents segmentation networks combined with the foreground attention boosting (FAB) module. Here we provide the implementation code on [COVID-19-CT-Seg](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#segmentation-task-2-learning-to-segment-covid-19-ct-scans-from-non-covid-19-ct-scans) dataset: 20 lung CT scans; Annotations include left lung, right lung and infections. And we evaluate our method on the infection segmentation task. For details, please refer to the segmentation task 1 of the [website](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#segmentation-task-2-learning-to-segment-covid-19-ct-scans-from-non-covid-19-ct-scans) , including the 5-fold cross-validation, dataset split file, preprocessing method, and the evaluation metrics.

| Task      | Training and testing                                         |
| --------- | ------------------------------------------------------------ |
| Infection | 5-fold cross validation<br />4 cases (20% for training)<br />16 cases (80% for testing) |

For [COVID-19-CT-Seg](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#segmentation-task-2-learning-to-segment-covid-19-ct-scans-from-non-covid-19-ct-scans) dataset, the baselines are based on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), which is a powerful segmentation method designed to deal with the dataset diversity found in the somain. It condenses and automates the keys decisions for designing a successful segmentation pipeline for any given dataset. In this repository, we apply the FAB module to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) without other special processing. One can refer to the [websit](https://github.com/MIC-DKFZ/nnUNet) for more implementation details of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). For convenience, we  provide one script  `demo.py` for easy testing. 

# Installation

All the experiments are implemented via PyTorch with one Nvidia RTX 2080ti GPU, Ubuntu system and Python 3. One can refer to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for more installation details or use the following process.

1. Install PyTorch (version 1.1.0)

2. Install Nvidia Apex by following instructions (after the installation of PyTorch):

   ```sh
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --no-cache-dir ./
   ```

3. Install dependent packages and nnU-Net:

   ```sh
   git clone https://github.com/Womcos/FAB-Net.git
   cd FAB-Net
   pip install -r requirements.txt
   pip install -e .
   ```

# Usage

The FAB-Net is based on the nnU-Net framework, and one can refer to the [website](https://github.com/MIC-DKFZ/nnUNet) for more implementation details.

### Testing with pre-trained model

Here we provide the  [pre-trained model](https://pan.baidu.com/s/17SuxN2lUoDTu9E3Nb_Fv_w) (password `9nhf` )and the testing code of the FAB-Net on the [COVID-19-CT-Seg](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#segmentation-task-2-learning-to-segment-covid-19-ct-scans-from-non-covid-19-ct-scans) dataset. For convenience, we provide a simple script `demo.py` for testing on the [COVID-19-CT-Seg](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#segmentation-task-2-learning-to-segment-covid-19-ct-scans-from-non-covid-19-ct-scans) dataset. The parameters to be used in the script `demo.py` are as follows:

```python
fold = '0'  ## '0','1','2','3','4'
test_folder = '.../COVID-19-CT-Seg/COVID-19-CT-Seg_20cases_edit1'
```

where `test_folder` is the folder of preprocessed raw images (10 cases from Coronacases are adjusted to lung window [-1250,250], and then normalized to [0, 255] ). The preprocessing code for raw images is given in the `./data_edit/demo.py`. The parameter `fold` represents the fold number of the pre-trained model according to the dataset split method of the [COVID-19-CT-Seg](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#segmentation-task-2-learning-to-segment-covid-19-ct-scans-from-non-covid-19-ct-scans) dataset. To use the pre-trained model for testing, one needs to set the `base` folder in `./nnunet/paths.py` as the address of the pre-trained model folder `.../nnunet2_COVID19_FAB` and set the `test_folder` as the folder of images for testing. After running `demo.py`, the test results will be saved in the folder `.../nnunet2_COVID19_FAB/test_results`. Then switch the fold from '0' to '4' to get results of all folds. For convenience, we provide the testing results of the validation data for all folds in the folder `.../validation_raw` of the [pre-trained model](https://pan.baidu.com/s/17SuxN2lUoDTu9E3Nb_Fv_w) (password `9nhf` ). One can easily evaluate the results without testing. In addition, the evaluation metrics are Dice similarity coefficient (DSC) and normalized surface Dice (NSD), and the python implementations are [here](http://medicaldecathlon.com/files/Surface_distance_based_measures.ipynb) which is given by the [COVID-19-CT-Seg](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#segmentation-task-2-learning-to-segment-covid-19-ct-scans-from-non-covid-19-ct-scans) dataset. 

# Results

The quantitative results of nnU-Net baseline and FAB-Net for infection segmentation task are presented as follows:

| Methods | Metrics | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg        |
| ------- | ------- | ------ | :----- | ------ | ------ | :----- | ---------- |
| nnU-Net | DSC     | 68.08% | 71.32% | 66.18% | 68.13% | 62.67% | 67.28%     |
|         | NSD     | 70.88% | 71.82% | 71.71% | 70.84% | 64.93% | 70.04%     |
| FAB-Net | DSC     | 73.59% | 74.10% | 73.67% | 72.12% | 66.30% | **71.95%** |
|         | NSD     | 76.05% | 74.47% | 80.23% | 75.42% | 68.73% | **74.98%** |

The quantitative results of nnU-Net baseline and FAB-Net for lung segmentation task are presented as follows:

| Methods | Lung  | Metrics | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg    |
| ------- | ----- | ------- | ------ | :----- | ------ | ------ | :----- | ------ |
| nnU-Net | Left  | DSC     | 84.88% | 80.28% | 87.14% | 88.44% | 88.33% | 85.82% |
|         |       | NSD     | 68.69% | 61.82% | 74.34% | 75.18% | 75.83% | 71.17% |
|         | Right | DSC     | 85.21% | 83.88% | 90.34% | 89.86% | 90.22% | 87.90% |
|         |       | NSD     | 70.55% | 68.25% | 78.45% | 78.45% | 78.31% | 74.80% |
| FAB-Net | Left  | DSC     | 88.61% | 86.50% | 91.76% |        |        |        |
|         |       | NSD     | 75.26% | 71.65% | 79.50% |        |        |        |
|         | Right | DSC     | 89.55% | 88.60% | 92.51% |        |        |        |
|         |       | NSD     | 76.25% | 74.87% | 80.47% |        |        |        |

The blank spaces are experiments under implementation.

# Questions

Please contact dxfeng@shu.edu.cn





