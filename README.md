<div align="center">
    <h2>
        STT: Building Extraction from Remote Sensing Images with Sparse Token Transformers
    </h2>
</div>
<br>

[//]: # (<div align="center">)

[//]: # (  <img src="resources/RSPrompter.png" width="800"/>)

[//]: # (</div>)
<br>
<div align="center">
  <a href="https://kychen.me/STT">
    <span style="font-size: 20px; ">Project Page</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.mdpi.com/2072-4292/13/21/4441">
    <span style="font-size: 20px; ">Paper</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/KyanChen/BuildingExtraction">
    <span style="font-size: 20px; ">HFSpace</span>
  </a>
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/KyanChen/STT)](https://github.com/KyanChen/STT)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/KyanChen/BuildingExtraction)

<br>
<br>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>


## Introduction

This repository is the code implementation of the paper [STT: Building Extraction from Remote Sensing Images with Sparse Token Transformers](https://www.mdpi.com/2072-4292/13/21/4441).

This branch has been tested with PyTorch 2.x and CUDA 12.1, supports Python 3.7+, and is compatible with most CUDA versions.

If you find this project helpful, please give us a star ⭐️. Your support is our biggest motivation.


## Updates

🌟 **2022.10.23** Released the STT project code.

🌟 **2022.10.25** Open-sourced the pre-trained models of WHU and INRIA datasets, you can find them on [Hugging Face Spaces](https://huggingface.co/KyanChen/BuildingExtraction/tree/main/Pretrain).

🌟 **2024.02.28** Reorganized the project.



## Table of Contents

- [Introduction](#introduction)
- [Updates](#updates)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [STT Model Training](#stt-model-training)
- [STT Model Testing](#stt-model-testing)
- [Citation](#citation)
- [License](#license)
- [Contact Us](#contact-us)

## Installation

### Dependencies

- Linux or Windows
- Python 3.7+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1
- CUDA 11.7 or higher, recommended 12.1

### Environment Installation


We recommend using Miniconda for installation. The following commands will create a virtual environment named `stt` and install PyTorch.


Note: If you are familiar with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow the steps below.

<details>


**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).


**Step 1**: Create a virtual environment named `stt` and activate it.

```shell
conda create -n stt python=3.10 -y
conda activate stt
```

**Step 2**: Install [PyTorch2.1.x](https://pytorch.org/get-started/locally/).

Linux/Windows:
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
Or

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 4**: Install other dependencies.

```shell
pip install -U pandas opencv-python tqdm scikit-image einops matplotlib
```


</details>


### Install STT



Download or clone the STT repository.

```shell
git clone git@github.com:KyanChen/STT.git
cd STT
```


## Dataset Preparation

<details>


### WHU Building Extraction Dataset


#### Data Download


- Image and label download address: [WHU](http://gpcv.whu.edu.cn/data/building_dataset.html).



#### Organization


You can also choose other sources to download the data, but you need to organize the dataset in the following format:

```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/WHU
├── train
│   ├── img
│   └── label
├── val
│   ├── img
│   └── label
└── test
    ├── img
    └── label
```


Note: In the project folder, we provide a folder named `Data`, which contains an example of the organization of the dataset.


### INRIA Building Extraction Dataset

#### Data Download

- Image and label download address: [INRIA](https://project.inria.fr/aerialimagelabeling/).
- You need to use the script [数据切分](Tools/CutImgSegWithLabel.py) to cut the data into a fixed size.

#### Organization

You can also choose other sources to download the data, but you need to organize the dataset in the above format.


### Other Datasets

If you want to use other datasets, you can refer to the above method to prepare the dataset.


### Dataset Configuration

- Use the script [GetTrainValTestCSV.py](Tools/GetTrainValTestCSV.py) to generate training, validation, and test csv files.
- Use the script [GetImgMeanStd.py](Tools/GetImgMeanStd.py) to get the mean and standard deviation of the training set images.

</details>

## STT Model Training

### Train File and Main Parameter Parsing

We provide the training script Train.py. Below we provide an analysis of some of the main parameters.

<details>

**Parameter Parsing**

- `line3`：os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'  Set the GPU to be used, generally no need to modify.
- `line17`：backbone: The type of the backbone network, generally no need to modify.
- `line22`：top_k_s: The number of spatially sparse tokens selected, generally no need to modify.
- `line23`：top_k_c: The number of channel sparse tokens selected, generally no need to modify.
- `line28`：BATCH_SIZE: The batch size of a single card, **needs to be modified according to the size of the memory**.
- `line31`：DATASET: The path of the training data list, **needs to be modified according to the path of the dataset**.
- `line32`：model_path: The save path of the model, generally no need to modify.
- `line33`：log_path: The save path of the log, generally no need to modify.
- `line35`：IS_VAL: Whether to verify, generally no need to modify.
- `line37`：VAL_DATASET: The path of the validation data list, **needs to be modified according to the path of the dataset**.
- `line39`：IS_TEST: Whether to test, generally no need to modify.
- `line40`：TEST_DATASET: The path of the test data list, **needs to be modified according to the path of the dataset**.
- `line45`：PRIOR_MEAN: The mean of the training set images, **needs to be modified according to the mean of the dataset**.
- `line46`：PRIOR_STD: The standard deviation of the training set images, **needs to be modified according to the standard deviation of the dataset**.
- `line53`：load_checkpoint_path: Whether to load the checkpoint, generally empty.
- `line55`：resume_checkpoint_path: Whether to resume training, generally empty.

</details>


### Single Card Training

```shell
CUDA_VISIBLE_DEVICES=0 python Train.py  # 0 is the GPU number used
```

### Multi-card Training

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python Train.py  # 0,1,2,3 are the GPU numbers used
```


## STT Model Testing

### Single Card Testing

We provide the training script Test.py. You need to modify `load_checkpoint_path` in `line47` of Test.py to the checkpoint file you want to use.



```shell
CUDA_VISIBLE_DEVICES=0 python Test.py  # 0 is the GPU number used
```

### Multi-card Testing

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python Test.py  # 0,1,2,3 are the GPU numbers used
```

Note: The output results will be saved in `log_path`.


## Citation


If you use the code or performance benchmarks of this project in your research, please refer to the following bibtex to cite TTP.

```
@Article{rs13214441,
AUTHOR = {Chen, Keyan and Zou, Zhengxia and Shi, Zhenwei},
TITLE = {Building Extraction from Remote Sensing Images with Sparse Token Transformers},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {21},
ARTICLE-NUMBER = {4441},
URL = {https://www.mdpi.com/2072-4292/13/21/4441},
ISSN = {2072-4292},
DOI = {10.3390/rs13214441}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Contact Us

If you have any other questions❓, please feel free to contact us 👬
