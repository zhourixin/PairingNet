# PairingNet: A Learning-based Pair-searching and -matching Network for Image Fragments

This repo contains official code and datasets for the ECCV 2024 paper [PairingNet: A Learning-based Pair-searching and -matching Network for Image Fragments](https://arxiv.org/abs/2312.08704).

## Installation
We tested on a server configured with Ubuntu 20.04, cuda 11.6 and gcc 9.3.0. Other similar configurations should also work.
1. Clone this repo:

```
git clone https://github.com/zhourixin/PairingNet.git
cd PairingNet
```

2. Install dependencies
```
conda env create --file requirments.yaml
conda activate PairingNet
```


This repo releases data introduced in our paper [PairingNet: A Learning-based Pair-searching and -matching Network for Image Fragments](https://arxiv.org/abs/2312.08704):

In this paper, we propose a learning-based image fragment pair-searching and -matching approach to solve the challenging restoration problem. Existing works use rule-based methods to match similar contour shapes or textures, which are always difficult to tune hyperparameters for extensive data and computationally time-consuming. Therefore, we propose a neural network that can effectively utilize neighbor textures with contour shape information to fundamentally improve performance. First, we employ a graph-based network to extract the local contour and texture features of fragments. Then, for the pair-searching task, we adopt a linear transformer-based module to integrate these local features and use contrastive loss to encode the global features of each fragment. For the pair-matching task, we design a weighted fusion module to dynamically fuse extracted local contour and texture features, and formulate a similarity matrix for each pair of fragments to calculate the matching score and infer the adjacent segment of contours. To faithfully evaluate our proposed network, we collect a real dataset and generate a simulated image fragment dataset through an algorithm we designed that tears complete images into irregular fragments. The experimental results show that our proposed network achieves excellent pair-searching accuracy, reduces matching errors, and significantly reduces computational time.

<div align="center">
<img src="./figure1.png" width="100%">
</div>
