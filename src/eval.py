# -*- coding: utf-8 -*-
import argparse
import time
import csv
import math
import numpy as np
import os

import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from custom_transforms_ae_gen2 import Normalize, CenterCrop, ArrayToTensor, Compose, Resize
from datasets.sequence_folders_ae_color import *

from itertools import cycle
from tqdm import tqdm
import scipy.misc
import itertools
from torchvision.utils import save_image
from path import Path

from utils import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from trainer import validate, validate_Make3D, validate_NYU
from AE_model_unet import *

parser = argparse.ArgumentParser(description='Depth AutoEncoder training on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=24, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-i', '--img_test', dest='img_test', action='store_true',
                    help='img test on validation set')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-metric', default='NYU_RtoD_gen2_ablation_study.csv', metavar='PATH',
                    help='csv where to save validation metric value')
parser.add_argument('--models_list_dir', type=str, default='./KITTI_AE_RtoD_trained_model_lr0002_color_uNet_gen2')
parser.add_argument('--result_dir', type=str, default='./AE_results')
parser.add_argument('--model_dir',type=str, default = './AE_trained_model_lr0000')
parser.add_argument('--gpu_num', type=str, default = "2")
parser.add_argument('--norm', type=str, default = "Batch")
parser.add_argument('--mode', type=str, default = "DtoD")
parser.add_argument('--height', type=int, default = 128)
parser.add_argument('--width', type=int, default = 416)
parser.add_argument('--dataset', type=str, default = "KITTI")
parser.add_argument('--evaluate', action='store_true', help='evaluate score')
parser.add_argument('--multi_test', action='store_true', help='test all of model in the dir')
parser.add_argument('--img_save', action='store_true', help='will save test set image')
parser.add_argument('--resize', action='store_true', help='will resize result image')

def main():
    args = parser.parse_args()
    print('=> number of GPU: ',args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print("=> information will be saved in {}".format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    img_H = args.height
    img_W = args.width
    
    training_writer = SummaryWriter(args.save_path)
        
    ########################################################################
    ######################   Data loading part    ##########################
    
    ## normalize -1 to 1 func
    normalize = Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    if args.dataset == 'NYU':
        valid_transform = Compose([
            CenterCrop(size=(img_H,img_W)), 
            ArrayToTensor(height=img_H, width=img_W),
            normalize
        ]) ### NYU valid transform ###
    elif args.dataset == 'KITTI':
        valid_transform = Compose([
            ArrayToTensor(height=img_H, width=img_W), 
            normalize
        ]) ### KITTI valid transform ###
    print("=> fetching scenes in '{}'".format(args.data))
    print("=> Dataset: ",args.dataset)

    if args.dataset == 'KITTI':
        print("=> test on Eigen test split")
        val_set = TestFolder(
            args.data,
            args = args,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            mode = args.mode
        )
    elif args.dataset == 'Make3D':
        val_set = Make3DFolder(
            args.data,
            args = args,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            mode = args.mode
        )
    elif args.dataset == 'NYU':
        val_set = NYUdataset(
            args.data,
            args = args,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            mode = args.mode
        )
    print('=> samples_num: {}- test'.format(len(val_set)))
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    cudnn.benchmark = True
    ###########################################################################
    
    ###################### setting model list #################################
    if args.multi_test is True:
        print("=> all of model tested")
        models_list_dir = Path(args.models_list_dir)
        models_list = sorted(models_list_dir.files('*.pkl'))
    else:
        print("=> just one model tested")
        models_list = [args.model_dir]


    ###################### setting Network part ###################

    print("=> creating base model")
    if args.mode == 'DtoD_test':
        print('- DtoD test')
        AE_DtoD = AutoEncoder_DtoD(norm=args.norm,input_dim=1,height=img_H,width=img_W)
        AE_DtoD = nn.DataParallel(AE_DtoD)
        AE_DtoD = AE_DtoD.cuda()
    elif args.mode == 'RtoD_test':
        print('- RtoD test')
        #AE_RtoD = AutoEncoder_Unet(norm=args.norm,height=img_H,width=img_W) #previous gradloss_mask model
        #AE_RtoD = AutoEncoder_2(norm=args.norm,input_dim=3,height=img_H,width=img_W) #current autoencoder_2 model
        AE_RtoD = AutoEncoder(norm=args.norm,height=img_H,width=img_W)
        AE_RtoD = nn.DataParallel(AE_RtoD)
        AE_RtoD = AE_RtoD.cuda()
    #############################################################################

    if args.evaluate is True:
        ############################ data log #######################################
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(val_loader), args.epoch_size), valid_size=len(val_loader))
        #logger.epoch_bar.start()
        with open(args.save_path/args.log_metric, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            if args.dataset == 'KITTI':
                writer.writerow(['Epoch','Abs_diff', 'Abs_rel','Sq_rel','a1','a2','a3','RMSE','RMSE_log'])
            elif args.dataset == 'Make3D':
                writer.writerow(['Epoch','Abs_diff', 'Abs_rel','log10','rmse'])
            elif args.dataset == 'NYU':
                writer.writerow(['Epoch','Abs_diff', 'Abs_rel','log10','a1','a2','a3','RMSE','RMSE_log'])
        ########################### Evaluating part #################################
        if args.mode == 'DtoD_test':
            test_model = AE_DtoD
            print("DtoD_test - eval 모드로 설정")
        elif args.mode == 'RtoD_test':
            test_model = AE_RtoD
            print("RtoD_test - eval 모드로 설정")

        test_len = len(models_list)
        print("=> Length of model list: ",test_len)

        for i in range(test_len):
            logger.reset_valid_bar()
            test_model.load_state_dict(torch.load(models_list[i]))
            test_model.eval()
            if args.dataset == 'KITTI':
                errors, min_errors, error_names = validate(args, val_loader, test_model, 0, logger,args.mode)
            elif args.dataset == 'Make3D':
                errors, min_errors, error_names = validate_Make3D(args, val_loader, test_model, 0, logger,args.mode)
            elif args.dataset == 'NYU':
                errors, min_errors, error_names = validate_NYU(args, val_loader, test_model, 0, logger,args.mode)
            for error, name in zip(errors, error_names):
                training_writer.add_scalar(name, error, 0)
            logger.valid_writer.write(' * RtoD_model: {}'.format(models_list[i]))
            #error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:len(error_names)], errors[0:len(errors)]))
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:len(error_names)], min_errors[0:len(errors)]))
            logger.valid_writer.write(' * Avg {}'.format(error_string))
            print("")        
            #error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:8], min_errors[0:8]))
            #logger.valid_writer.write(' * Avg {}'.format(error_string))
            logger.valid_bar.finish()
            with open(args.save_path/args.log_metric, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow(['%02d'%i]+['%.4f'%(min_errors[k]) for k in range(len(min_errors))])

        print(args.dataset," valdiation finish")
        ##  Test
        
        if args.img_save is False :
            print("--only Test mode finish--")
            return
    else:
        if args.mode == 'DtoD_test':
            test_model = AE_DtoD
            print("DtoD_test - eval 모드로 설정")
        elif args.mode == 'RtoD_test':
            test_model = AE_RtoD
            print("RtoD_test - eval 모드로 설정")
        test_model.load_state_dict(torch.load(models_list[0]))
        test_model.eval()
        print("=> No validation")
    
    k=0
    
    print("=> img save start")
    resize_ = Resize()
    for gt_data, rgb_data, filename in val_loader:
        if args.mode == 'RtoD' or args.mode == 'RtoD_test':
            gt_data = Variable(gt_data.cuda())
            final_AE_in = rgb_data.cuda()
        elif args.mode == 'DtoD' or args.mode == 'DtoD_test':
            rgb_data = Variable(rgb_data.cuda())
            final_AE_in = gt_data.cuda()
        final_AE_in = Variable(final_AE_in)
        with torch.no_grad():
            final_AE_depth = test_model(final_AE_in, istrain=False)
        img_arr = [final_AE_depth, gt_data, rgb_data]
        folder_name_list = ['/output_depth', '/ground_truth','/input_rgb']
        img_name_list = ['/final_AE_depth_', '/final_AE_gt_','/final_AE_rgb_']
        folder_iter = cycle(folder_name_list)
        img_name_iter = cycle(img_name_list)
        for img in img_arr:
            img_org = img.cpu().detach().numpy()
            folder_name = next(folder_iter)
            img_name = next(img_name_iter)
            result_dir = args.result_dir + folder_name
            for t in range(img_org.shape[0]):
                filename_ = filename[t]
                img = img_org[t]
                if img.shape[0] == 3:
                    img_ = np.empty([img_H,img_W,3])
                    img_[:,:,0] = img[0,:,:]
                    img_[:,:,1] = img[1,:,:]
                    img_[:,:,2] = img[2,:,:]
                    if args.resize is True:
                        img_ = resize_(img_,(384, 1248),'rgb')
                elif img.shape[0] == 1:
                    img_ = np.empty([img_H,img_W])
                    img_[:,:] = img[0,:,:]
                    if args.resize is True:
                        img_ = resize_(img_,(384, 1248),'depth')
                        img_ = img_[:,:,0]
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                scipy.misc.imsave(result_dir + img_name +'%05d.jpg'%(k+t),img_)
                #print(img_.shape)
                #print(filename_)
                #print(result_dir)
                #print(result_dir+filename_)
                #scipy.misc.imsave(result_dir + filename_ ,img_)
        k += img_org.shape[0]
    print("--Test image save finish--")
    return

if __name__ == "__main__":
    main()