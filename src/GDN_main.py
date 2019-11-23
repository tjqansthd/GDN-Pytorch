from option import args, parser
import csv
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from transform_list import *
from datasets.datasets_list import *
import os
from itertools import cycle
import scipy.misc
from utils import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from trainer import validate, validate_Make3D, train_AE_DtoD, train_AE_RtoD
from AE_model_unet import *

def main():
    print('=> number of GPU: ',args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print("=> information will be saved in {}".format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    img_H = args.height
    img_W = args.width
    if args.evaluate:
        args.epochs = 0
    training_writer = SummaryWriter(args.save_path)

    ########################################################################
    ######################   Data loading part    ##########################
    
    ## normalize -1 to 1 func
    normalize = Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    if args.dataset == "NYU":
        valid_transform = Compose([
            CenterCrop(size=(img_H,img_W)),
            ArrayToTensor(height=img_H, width=img_W), 
            normalize
        ]) ### NYU valid transform ###
    else:
        valid_transform = Compose([
            ArrayToTensor(height=img_H, width=img_W), 
            normalize
        ]) ### KITTI valid transform ###
    print("=> fetching scenes in '{}'".format(args.data))
    print("=> Dataset: ",args.dataset)

    if args.dataset == 'KITTI':
        train_transform = Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(),
            ArrayToTensor(height=img_H, width=img_W),
            normalize
        ])
        train_set = SequenceFolder(
            args.data, args = args, transform=train_transform,
            seed=args.seed, train=True, mode = args.mode)
        if args.real_test is False:
            print("=> test on validation set")
            '''
            val_set = SequenceFolder(
                args.data, args = args, transform=valid_transform,
                seed=args.seed, train=False, mode = args.mode)
            '''
            val_set = TestFolder(
                args.data, args = args, transform=valid_transform,
                seed=args.seed, train=False, mode = args.mode)
        else :
            print("=> test on Eigen test split")
            val_set = TestFolder(
                args.data, args = args, transform=valid_transform,
                seed=args.seed, train=False, mode = args.mode )
    elif args.dataset == 'Make3D':
        train_transform = Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(),
            ArrayToTensor(height=img_H, width=img_W),
            normalize
        ])
        train_set = Make3DFolder(
            args.data, args = args, transform=train_transform,
            seed=args.seed, train=True, mode = args.mode)
        val_set = Make3DFolder(
            args.data, args = args, transform=valid_transform,
            seed=args.seed, train=False, mode = args.mode)
    elif args.dataset == 'NYU':
        if args.mode == 'RtoD':
            print('RtoD transform created')
            train_transform = EnhancedCompose([
                Merge(),
                RandomCropNumpy(size=(251,340)),
                RandomRotate(angle_range=(-5, 5), mode='constant'),         
                Split([0, 3], [3, 4])
            ])
            train_transform_2 = EnhancedCompose([
                CenterCrop(size=(img_H,img_W)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2)), None],
                ArrayToTensor(height=img_H, width=img_W),
                normalize
            ])
        elif args.mode == 'DtoD':
            print('DtoD transform created')
            train_transform = EnhancedCompose([
                Merge(),
                RandomCropNumpy(size=(251,340)),
                RandomRotate(angle_range=(-4, 4), mode='constant'),         
                Split([0, 1])
            ])
            train_transform_2 = EnhancedCompose([
                CenterCrop(size=(img_H,img_W)),
                RandomHorizontalFlip(),
                ArrayToTensor(height=img_H, width=img_W),
                normalize
            ])
        train_set = NYUdataset(
            args.data, args = args, transform=train_transform, transform_2 = train_transform_2,
            seed=args.seed, train=True, mode = args.mode)
        val_set = NYUdataset(
            args.data, args = args, transform=valid_transform,
            seed=args.seed, train=False, mode = args.mode )
    #print('samples_num: {}  train scenes: {}'.format(len(train_set), len(train_set.scenes)))
    print('=> samples_num: {}  '.format(len(train_set)))
    print('=> samples_num: {}- test'.format(len(val_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)
    cudnn.benchmark = True
    ###########################################################################
    ###########################################################################

    ################################################################################
    ###################### Setting Network, Loss, Optimizer part ###################

    print("=> creating model")
    if args.mode == 'DtoD':
        print('- DtoD train')
        AE_DtoD = AutoEncoder_DtoD(norm=args.norm,input_dim=1,height=img_H,width=img_W)
        AE_DtoD = nn.DataParallel(AE_DtoD)
        AE_DtoD = AE_DtoD.cuda()
        #AE_DtoD.load_state_dict(torch.load(args.model_dir))
        print('- DtoD model is created')
        optimizer_AE = optim.Adam(AE_DtoD.parameters(), args.lr,[args.momentum,args.beta],eps=1e-08, weight_decay=5e-4)
        criterion_L2 = nn.MSELoss()
        criterion_L1 = nn.L1Loss()
    elif args.mode == 'RtoD':
        print('- RtoD train')
        AE_DtoD = AutoEncoder_DtoD(norm=args.norm,input_dim=1,height=img_H,width=img_W)
        AE_DtoD = nn.DataParallel(AE_DtoD)
        AE_DtoD = AE_DtoD.cuda()
        AE_DtoD.load_state_dict(torch.load(args.model_dir))
        AE_DtoD.eval()
        print('- pretrained DtoD model is created')
        AE_RtoD = AutoEncoder_2(norm=args.norm,input_dim=3,height=img_H,width=img_W)
        AE_RtoD = nn.DataParallel(AE_RtoD)
        AE_RtoD = AE_RtoD.cuda()
        #AE_RtoD.load_state_dict(torch.load(args.RtoD_model_dir))
        print('- RtoD model is created')
        optimizer_AE = optim.Adam(AE_RtoD.parameters(), args.lr,[args.momentum,args.beta],eps=1e-08, weight_decay=5e-4)
        criterion_L2 = nn.MSELoss()
        criterion_L1 = nn.L1Loss()
    elif args.mode == 'RtoD_single':
        print('- RtoD single train')
        AE_DtoD = None
        AE_RtoD = AutoEncoder_2(norm=args.norm,input_dim=3,height=img_H,width=img_W)
        AE_RtoD = nn.DataParallel(AE_RtoD)
        AE_RtoD = AE_RtoD.cuda()
        #AE_RtoD.load_state_dict(torch.load(args.RtoD_model_dir))
        print('- RtoD model is created')
        optimizer_AE = optim.Adam(AE_RtoD.parameters(), args.lr,[args.momentum,args.beta],eps=1e-08, weight_decay=5e-4)
        criterion_L2 = nn.MSELoss()
        criterion_L1 = nn.L1Loss()
    elif args.mode == 'DtoD_test':
        print('- DtoD test')
        AE_DtoD = AutoEncoder_DtoD(norm=args.norm,input_dim=1,height=img_H,width=img_W)
        AE_DtoD = nn.DataParallel(AE_DtoD)
        AE_DtoD = AE_DtoD.cuda()
        AE_DtoD.load_state_dict(torch.load(args.model_dir))
        print('- pretrained DtoD model is created')
    elif args.mode == 'RtoD_test':
        print('- RtoD test')
        AE_RtoD = AutoEncoder(norm=args.norm,height=img_H,width=img_W)
        #AE_RtoD = AutoEncoder_2(norm=args.norm,input_dim=3,height=img_H,width=img_W)
        AE_RtoD = nn.DataParallel(AE_RtoD)
        AE_RtoD = AE_RtoD.cuda()
        AE_RtoD.load_state_dict(torch.load(args.RtoD_model_dir))
        print('- pretrained RtoD model is created')

    #############################################################################
    #############################################################################

    ############################ data log #######################################
    if args.evaluate == True:
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
        logger.epoch_bar.start()
    elif args.evaluate == False:
        logger = None
    
    #logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    #logger.epoch_bar.start()
    if logger is not None:
        with open(args.save_path/args.log_summary, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss', 'validation_loss'])

        with open(args.save_path/args.log_full, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss_sum', 'output_loss', 'latent_loss'])
    
    #############################################################################

    ############################ Training part ##################################
    if args.mode == 'DtoD':
        loss = train_AE_DtoD(args,AE_DtoD,criterion_L2,criterion_L1, optimizer_AE,train_loader,val_loader,
            args.batch_size,args.epochs,args.lr,logger,training_writer)
        print('Final loss:',loss.item())
    elif args.mode == 'RtoD' or args.mode == 'RtoD_single':
        loss, output_loss, latent_loss = train_AE_RtoD(args,AE_RtoD, AE_DtoD, criterion_L2,criterion_L1, optimizer_AE, train_loader,
            val_loader,args.batch_size, args.epochs,args.lr,logger,training_writer)

    ########################### Evaluating part #################################
    if args.mode == 'DtoD_test':
        test_model = AE_DtoD
        print("DtoD_test - switch model to eval mode")
    elif args.mode == 'RtoD_test':
        test_model = AE_RtoD
        print("RtoD_test - switch model to eval mode")
    test_model.eval()
    if (logger is not None) and (args.evaluate == True):
        if args.dataset == 'KITTI':
            logger.reset_valid_bar()
            errors, min_errors, error_names = validate(args, val_loader, test_model, 0, logger,args.mode)
            error_length = 8
        elif args.dataset == 'Make3D':
            logger.reset_valid_bar()
            errors, min_errors, error_names = validate_Make3D(args, val_loader, test_model, 0, logger,args.mode)
            error_length = 4
        elif args.dataset == 'NYU':
            logger.reset_valid_bar()
            errors, min_errors, error_names = validate_NYU(args, val_loader, test_model, 0, logger,args.mode)
            error_length = 8
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, 0)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:error_length], errors[0:error_length]))
        logger.valid_writer.write(' * Avg {}'.format(error_string))
        print("")
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:error_length], min_errors[0:error_length]))
        logger.valid_writer.write(' * Avg {}'.format(error_string))
        logger.valid_bar.finish()
        print(args.dataset,"valdiation finish")
        
    ##  Test
    
    if args.img_save is False :
        print("--only Test mode finish--")
        return
    
    k=0
    
    for gt_data, rgb_data, _ in val_loader:
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
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            for t in range(img_org.shape[0]):
                img = img_org[t]
                if img.shape[0] == 3:
                    img_ = np.empty([img_H,img_W,3])
                    img_[:,:,0] = img[0,:,:]
                    img_[:,:,1] = img[1,:,:]
                    img_[:,:,2] = img[2,:,:]
                elif img.shape[0] == 1:
                    img_ = np.empty([img_H,img_W])
                    img_[:,:] = img[0,:,:]
                scipy.misc.imsave(result_dir + img_name +'%05d.jpg'%(k+t),img_)
        k += img_org.shape[0]


if __name__ == "__main__":
    main()