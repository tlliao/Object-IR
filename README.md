## Introduction
This is the official implementation for [Object-IR](https://arxiv.org/abs/xxx) (PR2025).

<p align="left">Tianli Liao<sup>1</sup>, Ran Wang<sup>1</sup>, Siqing Zhang<sup>1</sup>, Lei Li<sup>1</sup>, Guangen Liu<sup>1</sup>, Chenyang Zhao<sup>1</sup>, Heling Cao<sup>1</sup>, Peng Li<sup>2</sup></p>
<p align="left"><sup>1</sup>College of Information Science and Engineering, Henan University of Technology</p>
<p align="left"><sup>2</sup>Institute for Complexity Science, Henan University of Technology</p>


> ### Pipeline
> ![image](https://github.com/tlliao/Object-IR/blob/main/Object-IR.png)
Given any aspect ratio, we construct a rigid mesh for the output resolution and estimate the grid's motion via a CNN-based regression network.


<!-- ## 📝 Changelog -->

<!-- - [x] 2025.03.11: The paper of the arXiv version is online. -->

## Dataset (COCO)
The details of the dataset can be found in our paper. ([arXiv](https://arxiv.org/abs/xxx))

The dataset can be available at [Google Drive](https://drive.google.com/drive/folders/16EDGrKOLLwcMseOjpI7bCrv_aP1MYVcz?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1TKQAQ9zryUuU4uzTiswfHg) (Extraction code: 1234).

## Code

### Requirement
We implement Object-IR with one GPU of RTX3090. Refer to [requirements.txt](https://github.com/tlliao/Object-IR/blob/main/requirements.txt) for more details.

### Training

#### Step 1: Download the pretrained VGG19 model
Download [VGG-19](https://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search imagenet-vgg-verydeep-19 in this page and download imagenet-vgg-verydeep-19.mat. 

#### Step 2: Train the network
Modify the 'Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, we set 'ITERATIONS' to 100,000.

```
cd Codes/
python train.py
```

### Pre-trained model
The pre-trained models are available at [Google Drive](https://drive.google.com/drive/folders/1TuhQgD945MMnhmvnOwBS1LoLkYR1eetj?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1TTSbR4UYFL8f-nP3aGME7g) (extraction code: 1234). Please download them and put them in the 'model' folder.

### Test on the COCO dataset
Modify the test_path in Codes/test_online.py and run:
```
python test_online.py
```

### Test on arbitrary resolution images
Modify the 'Codes_for_Arbitrary_Resolution/constant.py'to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'Codes_for_Arbitrary_Resolution/inference.py'. 
Then, put the testing images into the folder 'Codes_for_Arbitrary_Resolution/other_dataset/' (including input and mask) and run:

```
cd Codes_for_Arbitrary_Resolution/
python inference.py
```


#### Calculate the metrics on the StabStitch-D dataset
Modify the test_path in Codes/test_metric.py and run:
```
python test_metric.py
```


## Citation
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
