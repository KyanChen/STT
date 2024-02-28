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
    <span style="font-size: 20px; ">项目主页</span>
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

[English](README.md) | 简体中文

</div>


## 简介

本项目仓库是论文 [STT: Building Extraction from Remote Sensing Images with Sparse Token Transformers](https://www.mdpi.com/2072-4292/13/21/4441) 的代码实现。

当前分支在PyTorch 2.x 和 CUDA 12.1 下测试通过，支持 Python 3.7+，能兼容绝大多数的 CUDA 版本。

如果你觉得本项目对你有帮助，请给我们一个 star ⭐️，你的支持是我们最大的动力。


## 更新日志

🌟 **2022.10.23** 发布了 STT 项目代码。

🌟 **2022.10.25** 开源了 WHU 和 INRIA 数据集的预训练模型，你可以在 [Hugging Face Spaces](https://huggingface.co/KyanChen/BuildingExtraction/tree/main/Pretrain) 上找到它们。

🌟 **2024.02.28** 重新整理了本项目。


[//]: # (## TODO)


## 目录

- [简介](#简介)
- [更新日志](#更新日志)
- [目录](#目录)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [STT模型训练](#STT模型训练)
- [STT模型测试](#STT模型测试)
- [引用](#引用)
- [开源许可证](#开源许可证)
- [联系我们](#联系我们)

## 安装

### 依赖项

- Linux 或 Windows
- Python 3.7+，推荐使用 3.10
- PyTorch 2.0 或更高版本，推荐使用 2.1
- CUDA 11.7 或更高版本，推荐使用 12.1

### 环境安装

我们推荐使用 Miniconda 来进行安装，以下命令将会创建一个名为 `stt` 的虚拟环境，并安装 PyTorch。

注解：如果你对 PyTorch 有经验并且已经安装了它，你可以直接跳转到下一小节。否则，你可以按照下述步骤进行准备。

<details>

**步骤 0**：安装 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)。

**步骤 1**：创建一个名为 `stt` 的虚拟环境，并激活它。

```shell
conda create -n stt python=3.10 -y
conda activate stt
```

**步骤 2**：安装 [PyTorch2.1.x](https://pytorch.org/get-started/locally/)。

Linux/Windows:
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
或者

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**步骤 4**：安装其他依赖项。

```shell
pip install -U pandas opencv-python tqdm scikit-image einops matplotlib
```


</details>

### 安装 STT

下载或克隆 STT 仓库即可。

```shell
git clone git@github.com:KyanChen/STT.git
cd STT
```

## 数据集准备

<details>

### WHU建筑物提取数据集

#### 数据下载

- 图片及标签下载地址： [WHU](http://gpcv.whu.edu.cn/data/building_dataset.html)。


#### 组织方式

你也可以选择其他来源进行数据的下载，但是需要将数据集组织成如下的格式：

```
${DATASET_ROOT} # 数据集根目录，例如：/home/username/data/WHU
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

注解：在项目文件夹中，我们提供了一个名为 `Data` 的文件夹，其中包含了上述数据集的组织方式的示例。

### INRIA建筑物提取数据集

#### 数据下载

- 图片及标签下载地址： [INRIA](https://project.inria.fr/aerialimagelabeling/).
- 需要利用脚本 [数据切分](Tools/CutImgSegWithLabel.py) 将数据切分为固定的大小。

#### 组织方式

你也可以选择其他来源进行数据的下载，但是需要将数据集组织成上述的格式。


### 其他数据集

如果你想使用其他数据集，可以参考上述方式来进行数据集的准备。

### 数据集配置

- 利用脚本 [生成图片列表](Tools/GetTrainValTestCSV.py) 来生成训练、验证和测试的 csv 文件。
- 利用脚本 [获取图片信息](Tools/GetImgMeanStd.py) 来获取训练集的图片的均值和标准差。

</details>

## STT模型训练

### Train 文件及主要参数解析

我们提供了训练脚本 Train.py。下面我们提供了一些主要参数的解析。

<details>

**参数解析**：

- `line3`：os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'  设置使用的 GPU，一般不需要修改。
- `line17`：backbone：主干网络的类型，一般不需要修改。
- `line22`：top_k_s：选择的空间稀疏的 token 数量，一般不需要修改。
- `line23`：top_k_c：选择的通道稀疏的 token 数量，一般不需要修改。
- `line28`：BATCH_SIZE：单卡的 batch size，**需要根据显存大小进行修改**。
- `line31`：DATASET：训练数据列表的路径，**需要根据数据集的路径进行修改**。
- `line32`：model_path：模型的保存路径，一般不需要修改。
- `line33`：log_path：日志的保存路径，一般不需要修改。
- `line35`：IS_VAL：是否进行验证，一般不需要修改。
- `line37`：VAL_DATASET：验证数据列表的路径，**需要根据数据集的路径进行修改**。
- `line39`：IS_TEST：是否进行测试，一般不需要修改。
- `line40`：TEST_DATASET：测试数据列表的路径，**需要根据数据集的路径进行修改**。
- `line45`：PRIOR_MEAN：训练集的图片的均值，**需要根据数据集的均值进行修改**。
- `line46`：PRIOR_STD：训练集的图片的标准差，**需要根据数据集的标准差进行修改**。
- `line53`：load_checkpoint_path：是否加载检查点，一般为空。
- `line55`：resume_checkpoint_path：是否断点续训，一般为空。

</details>


### 单卡训练

```shell
CUDA_VISIBLE_DEVICES=0 python Train.py  # 0 为使用的 GPU 编号
```

### 多卡训练

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python Train.py  # 0,1,2,3 为使用的 GPU 编号
```


## STT模型测试

### 单卡测试：

我们提供了训练脚本 Test.py。需要修改Test.py中`line47`的`load_checkpoint_path`为你想要使用的检查点文件。


```shell
CUDA_VISIBLE_DEVICES=0 python Test.py  # 0 为使用的 GPU 编号
```

### 多卡测试：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python Test.py  # 0,1,2,3 为使用的 GPU 编号
```

注解：输出的结果将会保存在 `log_path` 中。


## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 TTP。

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

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 联系我们

如果有其他问题❓，请及时与我们联系 👬
