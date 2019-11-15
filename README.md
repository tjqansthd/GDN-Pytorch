# GDN-Pytorch
This repository is a Pytorch implementation of the paper [**"Depth Estimation From a Single Image Using Guided Deep Network"**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8854079)

Minsoo Song and [Wonjun Kim](https://sites.google.com/site/kudcvlab)  
IEEE Access

When using this code in your research, please cite the following paper:  

Minsoo Song and Wonjun Kim, **"Depth estimation from a single image using guided deep network,"** **IEEE Access**, vol. 7, pp. 142595-142606, Dec. 2019.  

```
@ARTICLE{8854079,
author={M. {Song} and W. {Kim}},
journal={IEEE Access},
title={Depth Estimation From a Single Image Using Guided Deep Network},
year={2019},
volume={7},
pages={142595-142606},
doi={10.1109/ACCESS.2019.2944937},}
```

<p align="center"><img src='https://github.com/tjqansthd/GDN-Pytorch/blob/master/example/ex.png' width=800></p>  

## Model architecture
<p align="center"><img src='https://github.com/tjqansthd/GDN-Pytorch/blob/master/example/model_architecture.png' width=800></p>  

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

1st &nbsp;row: RGB input  
2nd row: Ground truth  
3rd row: Eigen et al.  
4th row: Godard et al.  
5th row: Kuznietsov et al.  
6th row: Ours

## Pretrained models
You can download pretrained color-to-depth model
* [Trained with KITTI](https://drive.google.com/drive/folders/10wFzSDdRK7nlNZXhSP3GGWuWFK_Ap7ur?usp=sharing)

## Demo (Single Image Prediction)
Demo Command Line:
```bash
############### Example of argument usage #####################
python depth_extract.py --gpu_num 0,1 --model_dir your/model/path/model.pkl 
## gpu_num is index of your gpu list.        ex) os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
```

### Try it on your own image!
1. Insert your example images(png, jpg) in ### GDN-pytorch/example/demo_input  
(Since our model was trained at 128 x 416 scale, we recommend resizing the images to the corresponding scale before running the demo.)
2. Specify the model directory, then run the demo. 

## Dataset
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)  
Official kitti dataset is available on the link.

We prepared the training data by referring to the method on [this link](https://github.com/josephdanielchang/unsupervised_learning_of_depth_PyTorch).

## Training
### Training method  
Depth_to_depth network training -> Color_to_depth network training(using pretrained depth_to_depth network)

* Depth_to_depth network training
```bash
python GDN_main.py ./your/dataset/path --epochs 50 --batch_size 20 --gpu_num 0,1,2,3 --mode DtoD
```
* Color_to_depth network training
```bash
python GDN_main.py ./your/dataset/path --epochs 50 --batch_size 20 --model_dir /your/pretrained/depth_to_depth/model/path --gpu_num 0,1,2,3 --mode RtoD
```
gpu_num is index of your gpu list. 

## Testing (Eigen split)
* Depth_to_depth network testing
```bash
python GDN_main.py /mnt/MS/AEdepth/data_backup --epochs 0 --batch_size 8 --evaluate --real_test --gpu_num 0,1,2,3 --model_dir /your/pretrained/depth_to_depth/model/path --mode DtoD_test --img_save
```

* Color_to_depth network testing
```bash
python GDN_main.py /mnt/MS/AEdepth/data_backup --epochs 0 --batch_size 8 --evaluate --real_test --gpu_num 0,1,2,3 --RtoD_model_dir /your/pretrained/color_to_depth/model/path --mode RtoD_test --img_save
```
gpu_num is index of your gpu list.  
if you want save your test result, using `--img_save`
