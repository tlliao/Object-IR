## Introduction
This is the official implementation for [Object-IR](https://arxiv.org/abs/2403.06378) (PR2026).

<p align="left">Tianli Liao<sup>1</sup>, Ran Wang<sup>1</sup>, Siqing Zhang<sup>1</sup>, Lei Li<sup>1</sup>, Guangen Liu<sup>1</sup>, Chenyang Zhao<sup>1</sup>, Heling Cao<sup>1</sup>, Peng Li<sup>2</sup></p>
<p align="left"><sup>1</sup>College of Information Science and Engineering, Henan University of Technology</p>
<p align="left"><sup>2</sup>Institute for Complexity Science, Henan University of Technology</p>


> ### Feature
> Nowadays, the videos captured from hand-held cameras are typically stable due to the advancements and widespread adoption of video stabilization in both hardware and software. Under such circumstances, we retarget video stitching to an emerging issue, warping shake, which describes the undesired content instability in non-overlapping regions especially when image stitching technology is directly applied to videos. To address it, we propose the first unsupervised online video stitching framework, named StabStitch, by generating stitching trajectories and smoothing them. 
![image](https://github.com/nie-lang/StabStitch/blob/main/fig.png)
The above figure shows the occurrence and elimination of warping shakes.


## 📝 Changelog

- [x] 2024.03.11: The paper of the arXiv version is online.
- [x] 2024.07.11: We have replaced the original arXiv version with the final camera-ready version.
- [x] 2024.07.11: The StabStitch-D dataset is available.
- [x] 2024.07.11: The inference code and pre-trained models are available.
- [x] 2024.07.12: We add a simple analysis of the limitations and prospects.

## Dataset (COCO)
The details of the dataset can be found in our paper. ([arXiv](https://arxiv.org/abs/2403.06378))

The dataset can be available at [Google Drive](https://drive.google.com/drive/folders/16EDGrKOLLwcMseOjpI7bCrv_aP1MYVcz?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1TKQAQ9zryUuU4uzTiswfHg)(Extraction code: 1234).

## Code
#### Requirement
We implement Object-IR with one GPU of RTX3090. Refer to [environment.yml](https://github.com/nie-lang/StabStitch/blob/main/environment.yml) for more details.

#### Pre-trained model
The pre-trained models are available at [Google Drive](https://drive.google.com/drive/folders/1TuhQgD945MMnhmvnOwBS1LoLkYR1eetj?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1TTSbR4UYFL8f-nP3aGME7g) (extraction code: 1234). Please download them and put them in the 'model' folder.

#### Test on the COCO dataset
Modify the test_path in Codes/test_online.py and run:
```
python test_online.py
```
Then, a folder named 'result' will be created automatically to store the stitched videos.

About the TPS warping function, we set two modes to warp frames as follows:
* 'FAST' mode: It uses F.grid_sample to implement interpolation. It's fast but may produce thin black boundaries.
* 'NORMAL' mode: It uses our implemented interpolation function. It's a bit slower but avoid the black boundaries.

You can change the mode [here](https://github.com/nie-lang/StabStitch/blob/0c3665377e8bb76e062d5276cda72a7c7f0fab5b/Codes/test_online.py#L127).


#### Calculate the metrics on the StabStitch-D dataset
Modify the test_path in Codes/test_metric.py and run:
```
python test_metric.py
```


## Meta
If you have any questions about this project, please feel free to drop me an email.

Tianli Liao -- tianli.liao@haut.edu.cn
```
@inproceedings{liao2026object-ir,
  title={Object-IR: Leveraging Object Consistency and Mesh Deformation for Self-Supervised Image Retargeting},
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Zhang, Yun and Liu, Shuaicheng and Ai, Rui and Zhao, Yao},
  booktitle={European Conference on Computer Vision},
  pages={390--407},
  year={2024},
  organization={Springer}
}
```
