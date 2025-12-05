## 🔍 Introduction
This is the official implementation for [Object-IR](https://arxiv.org/abs/2510.27236) (PR2026).

<p align="left">Tianli Liao<sup>1</sup>, Ran Wang<sup>1</sup>, Siqing Zhang<sup>1</sup>, Lei Li<sup>1</sup>, Guangen Liu<sup>1</sup>, Chenyang Zhao<sup>1</sup>, Heling Cao<sup>1</sup>, Peng Li<sup>2</sup></p>
<p align="left"><sup>1</sup>College of Information Science and Engineering, Henan University of Technology</p>
<p align="left"><sup>2</sup>Institute for Complexity Science, Henan University of Technology</p>


> ### Pipeline
> ![image](https://github.com/tlliao/Object-IR/blob/main/Object-IR.png)
Given any aspect ratio, we construct a rigid mesh for the output resolution and estimate the grid's motion via a CNN-based regression network.


## 📝 Changelog 

- [x] 2025.12.05: The training code, dataset and pretrained model are online.
- [x] 2025.11.03: The paper of the arXiv version is online.

<!-- ## Dataset (COCO)
The details of the dataset can be found in our paper. ([arXiv](https://arxiv.org/abs/xxx))

The dataset can be available at [Google Drive]() or [Baidu Cloud]() (Extraction code: xxxx). -->

## 🐍 Code

### Requirement
* python 3.8.5
* numpy 1.24.4
* pytorch 2.4.1
* tensorboard 2.13.0

We implement Object-IR with one GPU of RTX3090. Refer to [environment.yml](https://github.com/tlliao/Object-IR/blob/main/environment.yml) for more details.

### Training
#### Step 1: Download the dataset
First download the [COCO](https://drive.google.com/drive/folders/1zuGA-WQNFFn-TpbYG-UqqzUQAv_9DUTr) dataset (or in [Baidu Cloud](https://pan.baidu.com/s/1sIPU7gTvArKUjpqEDHNfbg), extraction code: 1205). Unzip and put the dataset in the "Data/" directory.

#### Step 2: Start training

Run the following command to start the training:
```
python train.py
```
The trained model will be saved in the "model/" directory.

### Pre-trained model
The pre-trained model are available at [Google Drive](https://drive.google.com/drive/folders/1eXit_ip9N04UjGysH5SDBY6dOqF_-ikB) or [Baidu Cloud](https://pan.baidu.com/s/1sIPU7gTvArKUjpqEDHNfbg) (extraction code: 1205). Please download them and put them in the 'model' folder.

### Testing

#### Step 1: Generate the warped images
Set the train/test dataset path in Codes/test_output.py and run:
```
python test_output.py
```

#### Step 2: Calculate the distortion error
Set the test dataset path in Codes/test.py and run
```
python test.py
```


## 📚 Citation
If you have any questions, please don't hesitate to contact me.

Tianli Liao -- tianli.liao@haut.edu.cn
```
@article{liao2026object-ir,
title = {Object-IR: Leveraging object consistency and mesh deformation for self-supervised image retargeting},
author = {Tianli Liao and Ran Wang and Siqing Zhang and Lei Li and Guangen Liu and Chenyang Zhao and Heling Cao and Peng Li},
journal = {Pattern Recognition},
volume = {172},
pages = {112651},
year = {2026},
}
```
