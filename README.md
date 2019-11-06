# GDN-Pytorch
This repository is a Pytorch implementation of the paper ["Depth Estimation From a Single Image Using Guided Deep Network"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8854079)

Codes will be coming soon!  

<p align="center"><img src='https://github.com/tjqansthd/GDN-Pytorch/blob/master/example/ex1.png' width=800></p>

Minsoo Song and [Wonjun Kim](https://sites.google.com/site/kudcvlab)  
IEEE Access

## Requirements

* Python >= 3.5
* Pytorch 0.4.0
* Ubuntu 16.04
* CUDA 8 (if CUDA available)
* cuDNN (if CUDA available)

## Video
<img src='https://github.com/tjqansthd/GDN-Pytorch/blob/master/example/rgb1.gif' width=400>   <img src='https://github.com/tjqansthd/GDN-Pytorch/blob/master/example/out2.gif' width=400>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RGB input&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Estimated depth map  

## Results
<p align="center"><img src='https://github.com/tjqansthd/GDN-Pytorch/blob/master/example/result1.png' width=1000></p>

<p align="center"><img src='https://github.com/tjqansthd/GDN-Pytorch/blob/master/example/result2.png' width=1000></p>

<p align="center"><img src='https://github.com/tjqansthd/GDN-Pytorch/blob/master/example/result3.png' width=1000></p>

1st rows: input RGB color images. 2nd rows: ground truth. 3rd rows: Eigen et al. 4th rows: Godard et al. 5th rows: Kuznietsov et al. 6th  rows: Ours
