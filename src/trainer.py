import numpy as np
from utils import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
import time
from calculate_error import *
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import csv
import os
import scipy.misc
from tqdm import tqdm
from path import Path


def validate(args, val_loader, model, epoch, logger,mode='DtoD'):
    ##global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3','rmse','rmse_log']
    errors = AverageMeter(i=len(error_names))
    min_errors = AverageMeter(i = len(error_names))
    min_errors_list = []

    abs_diff_tot, abs_rel_tot, sq_rel_tot, a1_tot, a2_tot, a3_tot, rmse_tot, rmse_log_tot = [],[],[],[],[],[],[],[]
    abs_diff_sum, abs_rel_sum, sq_rel_sum, a1_sum, a2_sum, a3_sum, rmse_sum, rmse_log_sum = 0,0,0,0,0,0,0,0

    # switch to evaluate mode
    #model.eval()
    print("mode: ",args.mode)
    end = time.time()
    logger.valid_bar.update(0)
    for i, (depth, img, depth_np) in enumerate(val_loader):
        img = img.cuda()
        depth = depth.cuda()
        depth_np = depth_np.cuda()
        # compute output
        if mode == 'RtoD' or mode == 'RtoD_test':
            input_img = img
        elif mode == 'DtoD' or mode == 'DtoD_test':
            input_img = depth
        with torch.no_grad():
            output_depth = model(input_img, istrain=False)
        err_result = compute_errors(depth_np, depth, output_depth,crop=True)
        errors.update(err_result)
        abs_diff_tot.append(err_result[0])
        abs_rel_tot.append(err_result[1])
        sq_rel_tot.append(err_result[2])
        a1_tot.append(err_result[3])
        a2_tot.append(err_result[4])
        a3_tot.append(err_result[5])
        rmse_tot.append(err_result[6])
        rmse_log_tot.append(err_result[7])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    
    logger.valid_bar.update(len(val_loader))
    sorted_abs_diff = sorted(abs_diff_tot)
    #min_len = 72
    min_len = (len(sorted_abs_diff))
    print("scene length: ",min_len)
    print("sorted_abs_diff length: ",len(sorted_abs_diff))
    for i in range(min_len):
        sort_idx = abs_diff_tot.index(sorted_abs_diff[i])
        abs_diff_sum += sorted_abs_diff[i]
        abs_rel_sum += abs_rel_tot[sort_idx]
        sq_rel_sum += sq_rel_tot[sort_idx]
        a1_sum += a1_tot[sort_idx]
        a2_sum += a2_tot[sort_idx]
        a3_sum += a3_tot[sort_idx]
        rmse_sum += rmse_tot[sort_idx]
        rmse_log_sum += rmse_log_tot[sort_idx]
    min_errors_list.append(abs_diff_sum/min_len)
    min_errors_list.append(abs_rel_sum/min_len)
    min_errors_list.append(sq_rel_sum/min_len)
    min_errors_list.append(a1_sum/min_len)
    min_errors_list.append(a2_sum/min_len)
    min_errors_list.append(a3_sum/min_len)
    min_errors_list.append(rmse_sum/min_len)
    min_errors_list.append(rmse_log_sum/min_len)
    min_errors.update(min_errors_list)

    return errors.avg,min_errors.avg,error_names

def validate_in_test(args, val_loader, model, DtoD_model,epoch, logger,mode='DtoD',crop_mask = None,criterion_L2=None):
    ##global device
    valid_cnt = 0
    total_loss = torch.tensor(0.).cuda()
    # switch to evaluate mode
    model.eval()
    print("mode: ",args.mode)

    for i, (depth, img, depth_np) in enumerate(val_loader):
        valid_cnt = valid_cnt + 1
         # data loading time
        if logger is not None:
            data_time.update(time.time() - end)
        # get the inputs
        inputs = img
        depths = depth
        if args.dataset != "KITTI":
            depth_np = None
        # If gt_data_2 is None ==> NYU dataset!
        if depth_np is not None:
            sparse_depths = depth_np.cuda()
            sparse_depths = Variable(sparse_depths)

        origin = depths
        inputs = inputs.cuda()
        depths = depths.cuda()

        # wrap them in Variable
        inputs, depths = Variable(inputs), Variable(depths)
        
        ########################################
        ### Train the AutoEncoder (Generator) ###
        ########################################
        
        '''AutoEncoder loss'''
        with torch.no_grad():
            outputs = model(inputs,istrain=False)

        if args.mode != 'RtoD_single':
            with torch.no_grad():
                ft_map1_tar, ft_map2_tar, ft_map3_tar, ft_map4_tar, _, _, _, _ = DtoD_model(depths, istrain=True)

        if args.mode != 'RtoD_single':
            with torch.no_grad():
                ft_map1, ft_map2, ft_map3, ft_map4, _, _, _, _ = DtoD_model(outputs, istrain=True)
        
        # masking valied area
        if depth_np is not None:
            valid_mask = sparse_depths > -1
            valid_mask = valid_mask[:,0,:,:].unsqueeze(1)
            if(crop_mask.size(0) != valid_mask.size(0)):
                crop_mask = crop_mask[0:valid_mask.size(0),:,:,:]

        diff = outputs - depths
        diff_abs = torch.abs(diff)
        diff_2 = torch.pow(outputs-depths,2)
        c = 0.2*torch.max(diff_abs.detach())         
        mask2 = torch.gt(diff_abs.detach(),c)
        diff_abs[mask2] = (diff_2[mask2]+(c*c))/(2*c)
        if depth_np is not None:
            diff_abs[~crop_mask] = 0.1*diff_abs[~crop_mask]      
            diff_abs[crop_mask&(~valid_mask)] = 0.3*diff_abs[crop_mask&(~valid_mask)]
        output_loss = 3*diff_abs.mean()
        
        diff2_clone = diff_2.clone().detach()
        rmse_loss = torch.sqrt(diff2_clone.mean())

        ################# BerHu Loss #########################
        latent_loss = torch.tensor(0.).cuda()
        if args.mode != 'RtoD_single':
            latent1 = criterion_L2(ft_map1, ft_map1_tar.detach())
            latent2 = 2.5*criterion_L2(ft_map2, ft_map2_tar.detach())
            latent3 = 14*criterion_L2(ft_map3, ft_map3_tar.detach())
            latent4 = 12*criterion_L2(ft_map4, ft_map4_tar.detach()) 
            #print("latent1 : ",latent1.item(),"latent2 : ",latent2.item(),"latent3 : ",latent3.item(),"latent4 : ",latent4.item())
            latent_loss = 1.5*((latent1 + latent2 + latent3 + latent4)/4) 
        
        ################# Latent Loss #########################

        gradient_loss = imgrad_loss(outputs, depths) ## for kitti
        ##gradient_loss = 3.5* imgrad_loss(outputs, depths) ## for NYU
        ################# gradient loss #######################
        grad_latent_loss = torch.tensor(0.).cuda()
        if args.mode != 'RtoD_single':
            grad_latent1 = imgrad_loss(ft_map1, ft_map1_tar.detach())
            grad_latent2 = 1.5*imgrad_loss(ft_map2, ft_map2_tar.detach())
            grad_latent3 = 4*imgrad_loss(ft_map3, ft_map3_tar.detach()) ##for kitti
            ##grad_latent3 = 2*imgrad_loss(ft_map3, ft_map3_tar.detach()) ##for NYU
            grad_latent4 = 2.3*imgrad_loss(ft_map4, ft_map4_tar.detach())
            #print("g_latent1 : ",grad_latent1.item(),"g_latent2 : ",grad_latent2.item(),"g_latent3 : ",grad_latent3.item(),"g_latent4 : ",grad_latent4.item())
            ##grad_latent_loss = ((grad_latent1 + grad_latent2 + grad_latent3 + grad_latent4)/4.0) ## for kitti
            grad_latent_loss = ((grad_latent1 + grad_latent2 + grad_latent3 + grad_latent4)/4.0) ## for NYU
        ################# gradient latent loss ################

        grad_loss = (gradient_loss + grad_latent_loss)
        ################# gradient total loss ################
        
        
        depth_smoothness_tot = 0.1*depth_smoothness(outputs,inputs) 
        depth_smoothness_loss = torch.mean(torch.abs(depth_smoothness_tot))
        
        ################# smoothness loss ######################
        loss = output_loss + latent_loss + grad_loss + depth_smoothness_loss
        total_loss = total_loss + loss
    total_loss = total_loss / valid_cnt
    total_loss = total_loss.item()
    rmse_loss = rmse_loss.item()
    # turn back to train mode
    model.train()
    return total_loss, rmse_loss

def validate_NYU(args, val_loader, model, epoch, logger,mode='DtoD'):
    ##global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'log10', 'a1', 'a2', 'a3','rmse','rmse_log']
    errors = AverageMeter(i=len(error_names))
    min_errors = AverageMeter(i = len(error_names))
    min_errors_list = []

    abs_diff_tot, abs_rel_tot, log10_tot, a1_tot, a2_tot, a3_tot, rmse_tot, rmse_log_tot = [],[],[],[],[],[],[],[]
    abs_diff_sum, abs_rel_sum, log10_sum, a1_sum, a2_sum, a3_sum, rmse_sum, rmse_log_sum = 0,0,0,0,0,0,0,0

    # switch to evaluate mode
    #model.eval()
    print("mode: ",args.mode)
    end = time.time()
    logger.valid_bar.update(0)
    for i, (depth, img, _) in enumerate(val_loader):
        img = img.cuda()
        depth = depth.cuda()
        # compute output
        if mode == 'RtoD' or mode == 'RtoD_test':
            input_img = img
        elif mode == 'DtoD' or mode == 'DtoD_test':
            input_img = depth
        with torch.no_grad():
            output_depth = model(input_img, istrain=False)
        err_result = compute_errors_NYU(depth, output_depth,crop=True)
        errors.update(err_result)
        abs_diff_tot.append(err_result[0])
        abs_rel_tot.append(err_result[1])
        log10_tot.append(err_result[2])
        a1_tot.append(err_result[3])
        a2_tot.append(err_result[4])
        a3_tot.append(err_result[5])
        rmse_tot.append(err_result[6])
        rmse_log_tot.append(err_result[7])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    
    logger.valid_bar.update(len(val_loader))
    sorted_abs_diff = sorted(abs_diff_tot)
    #min_len = 72
    min_len = (len(sorted_abs_diff))
    print("scene length: ",min_len)
    print("sorted_abs_diff length: ",len(sorted_abs_diff))
    for i in range(min_len):
        sort_idx = abs_diff_tot.index(sorted_abs_diff[i])
        abs_diff_sum += sorted_abs_diff[i]
        abs_rel_sum += abs_rel_tot[sort_idx]
        log10_sum += log10_tot[sort_idx]
        a1_sum += a1_tot[sort_idx]
        a2_sum += a2_tot[sort_idx]
        a3_sum += a3_tot[sort_idx]
        rmse_sum += rmse_tot[sort_idx]
        rmse_log_sum += rmse_log_tot[sort_idx]
    min_errors_list.append(abs_diff_sum/min_len)
    min_errors_list.append(abs_rel_sum/min_len)
    min_errors_list.append(log10_sum/min_len)
    min_errors_list.append(a1_sum/min_len)
    min_errors_list.append(a2_sum/min_len)
    min_errors_list.append(a3_sum/min_len)
    min_errors_list.append(rmse_sum/min_len)
    min_errors_list.append(rmse_log_sum/min_len)
    min_errors.update(min_errors_list)

    return errors.avg,min_errors.avg,error_names

def validate_Make3D(args, val_loader, model, epoch, logger,mode='DtoD'):
    ##global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']
    errors = AverageMeter(i=len(error_names))
    min_errors = AverageMeter(i = len(error_names))
    min_errors_list = []

    abs_diff_tot, abs_rel_tot, ave_log10_tot, rmse_tot = [],[],[],[]
    abs_diff_sum, abs_rel_sum, ave_log10_sum, rmse_sum = 0,0,0,0

    # switch to evaluate mode
    #model.eval()
    print("mode: ",args.mode)
    end = time.time()
    logger.valid_bar.update(0)
    for i, (depth, img, depth_np) in enumerate(val_loader):
        img = img.cuda()
        depth = depth.cuda()
        depth_np = depth_np.cuda()
        # compute output
        if mode == 'RtoD' or mode == 'RtoD_test':
            input_img = img
        elif mode == 'DtoD' or mode == 'DtoD_test':
            input_img = depth
        with torch.no_grad():
            output_depth = model(input_img, istrain=False)
        err_result = compute_errors_Make3D(depth_np, depth, output_depth)
        errors.update(err_result)
        abs_diff_tot.append(err_result[0])
        abs_rel_tot.append(err_result[1])
        ave_log10_tot.append(err_result[2])
        rmse_tot.append(err_result[3])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    
    logger.valid_bar.update(len(val_loader))
    sorted_abs_diff = sorted(abs_diff_tot)
    #min_len = 72
    min_len=(len(sorted_abs_diff))
    print("scene length: ",min_len)
    print("sorted_abs_diff length: ",len(sorted_abs_diff))
    for i in range(min_len):
        sort_idx = abs_diff_tot.index(sorted_abs_diff[i])
        abs_diff_sum += sorted_abs_diff[i]
        abs_rel_sum += abs_rel_tot[sort_idx]
        ave_log10_sum += ave_log10_tot[sort_idx]
        rmse_sum += rmse_tot[sort_idx]
    min_errors_list.append(abs_diff_sum/min_len)
    min_errors_list.append(abs_rel_sum/min_len)
    min_errors_list.append(ave_log10_sum/min_len)
    min_errors_list.append(rmse_sum/min_len)
    min_errors.update(min_errors_list)

    return errors.avg,min_errors.avg,error_names

def train_AE_DtoD(args,model,criterion_L2,criterion_L1, optimizer, dataset_loader,val_loader, batch_size, n_epochs,lr,logger, train_writer):
    global n_iter, best_error
    print("Training for %d epochs..." % n_epochs)
    num = 0
    model_num = 0
    data_iter = iter(dataset_loader)
    depth_fixed, rgb_fixed, _ = next(data_iter)
    depth_fixed = depth_fixed.cuda()
    rgb_fixed = rgb_fixed.cuda()

    predicted_dirs = './' + args.dataset + '_AE_DtoD_predicted_lr000%d_color_uNet_gen2_nogradf'%(lr*100000)
    result_dirs = './' + args.dataset + '_AE_DtoD_feat_result_lr000%d_color_uNet_gen2_nogradf/out'%(lr*100000)
    result_gt_dirs = './' + args.dataset + '_AE_DtoD_feat_result_lr000%d_color_uNet_gen2_nogradf/gt'%(lr*100000)  
    save_dir = './' + args.dataset + '_AE_DtoD_trained_model_lr000%d_color_uNet_gen2_nogradf'%(lr*100000)
    if((args.local_rank + 1)%4 == 0):
        if not os.path.exists(predicted_dirs):
            os.makedirs(predicted_dirs)
        if not os.path.exists(result_dirs):
            os.makedirs(result_dirs)
        if not os.path.exists(result_gt_dirs):
            os.makedirs(result_gt_dirs)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    H = depth_fixed.shape[2]
    W = depth_fixed.shape[3]
    num_sample_list = [16,64,64,64,64,64,16]
    figsize_x_list = [14,16,10,10,10,16,14]
    figsize_y_list = [7,8,5,5,5,8,7]

    if args.dataset != 'NYU':
        d_range_list = [4,2,2,4,2,2,4]
        ftmap_height_list = [H, int(H/2), int(H/8), int(H/16), int(H/8), int(H/2), H]
        ftmap_width_list = [W, int(W/2), int(W/8), int(W/16), int(W/8), int(W/2), W]
    else:
        d_range_list = [4,2,4,2,4,2,4]
        ftmap_height_list = [H, int(H/2), int(H/4), int(H/8), int(H/16), int(H/2), H]
        ftmap_width_list = [W, int(W/2), int(W/4), int(W/8), int(W/16), int(W/2), W]

    test_loss_dir = Path(args.save_path)
    test_loss_dir_rmse = str(test_loss_dir/'test_rmse_list.txt')
    test_loss_dir = str(test_loss_dir/'test_loss_list.txt')
    train_loss_dir = Path(args.save_path)
    train_loss_dir_rmse = str(train_loss_dir/'train_rmse_list.txt')
    train_loss_dir = str(train_loss_dir/'train_loss_list.txt')
    
    loss_list = []
    rmse_list = []
    train_loss_list = []
    train_rmse_list = []
    num_cnt = 0
    train_loss_cnt = 0

    if args.dataset == "KITTI":
        y1,y2 = int(0.40810811 * depth_fixed.size(2)), int(0.99189189 * depth_fixed.size(2))
        x1,x2 = int(0.03594771 * depth_fixed.size(3)), int(0.96405229 * depth_fixed.size(3))    ### Crop used by Garg ECCV 2016
        '''
        y1,y2 = int(0.3324324 * depth_fixed.size(2)), int(0.91351351 * depth_fixed.size(2))
        x1,x2 = int(0.0359477 * depth_fixed.size(3)), int(0.96405229 * depth_fixed.size(3))     ### Crop used by Godard CVPR 2017   
        '''
        print(" - valid y range: %d ~ %d"%(y1,y2))
        print(" - valid x range: %d ~ %d"%(x1,x2))
    for epoch in tqdm(range(n_epochs)):
        if args.dataset == "KITTI":
            crop_mask = depth_fixed != depth_fixed
            #print('crop_mask size: ',crop_mask.size())
            crop_mask[:,:,y1:y2,x1:x2] = 1
        if logger is not None:
            logger.epoch_bar.update(epoch)
            ####################################### one epoch training #############################################
            logger.reset_train_bar()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(precision=4)
        ################ train mode ####################
        model.train()
        ################################################
        end = time.time()
        if logger is not None:
            logger.train_bar.update(0)
        for i, (gt_data, _, gt_data_2) in enumerate(dataset_loader):
            # data loading time
            if logger is not None:
                data_time.update(time.time() - end)
            
            # get the inputs and wrap them in Variable
            origin = gt_data
            depths = gt_data.cuda()
            depths = Variable(depths)
            if args.dataset != "KITTI":
                gt_data_2 = None                # If gt_data_2 is None ==> NYU dataset!
            else:
                sparse_depths = gt_data_2.cuda()
                sparse_depths = Variable(sparse_depths)
            
            ##################### Feedforward ########################
            outputs = model(depths,istrain=False)
            ##########################################################

            '''AutoEncoder loss'''
            ########################## BerHu Loss ###################################
            # masking valied area
            if gt_data_2 is not None:
                valid_mask = sparse_depths > -1
                valid_mask = valid_mask[:,0,:,:].unsqueeze(1)
                if(crop_mask.size(0) != valid_mask.size(0)):
                    crop_mask = crop_mask[0:valid_mask.size(0),:,:,:]

            diff = outputs - depths
            diff_abs = torch.abs(diff)
            diff_2 = torch.pow(outputs-depths,2)
            c = 0.2*torch.max(diff_abs.detach())         
            mask2 = torch.gt(diff_abs.detach(),c)
            diff_abs[mask2] = (diff_2[mask2]+(c*c))/(2*c)
            if gt_data_2 is not None:
                diff_abs[~crop_mask] = 0.1*diff_abs[~crop_mask]      
                diff_abs[crop_mask&(~valid_mask)] = 0.3*diff_abs[crop_mask&(~valid_mask)]
            output_loss = 3*diff_abs.mean()
            ##########################################################################

            ####################### gradient loss ############################
            #gradient_loss = imgrad_loss(outputs, depths.detach())              # KITTI loss weight
            gradient_loss = 3*imgrad_loss(outputs, depths.detach())             # NYU loss weight
            ##################################################################

            loss = output_loss + gradient_loss
            #print("total_loss: %5f, output_loss: %5f, gradient_loss: %5f"%(loss.item(),output_loss.item(),gradient_loss.item()))
            
            if logger is not None:
                if i > 0 and n_iter % args.print_freq == 0:
                    train_writer.add_scalar('total_loss', loss.item(), n_iter)
                # record loss and EPE
                losses.update(loss.item(), args.batch_size)

            ###### zero the parameter gradients and backward & optimize ######
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ##################################################################
            
            # measure elapsed time
            if logger is not None:
                batch_time.update(time.time() - end)
                end = time.time()

                with open(args.save_path/args.log_full, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow([loss.item()])
                logger.train_bar.update(i+1)
                if i % args.print_freq == 0:
                    logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
                n_iter += 1
            if i >= args.epoch_size - 1:
                break
            ### KITTI's learning decay ###
            '''
            if (epoch>2):
                if ((i+1) % 2200 == 0):
                    if (lr < 0.00002):
                        lr -= (lr / 100)
                    else :
                        lr -= (lr / 60)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decayed learning rates, lr: {}'.format(lr))
            '''
            ### NYU's learning decay ###
            if (epoch>5):
                if ((i+1) % 1900 == 0):
                    if (lr < 0.00002):
                        lr -= (lr / 100)
                    else :
                        lr -= (lr / 25)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decayed learning rates, lr: {}'.format(lr))
            if ((i+1) % 50 == 0):
                print("epoch: %d,  %d/%d"%(epoch+1,i+1,args.epoch_size))
                print("total_loss: %5f, output_loss: %5f, gradient_loss: %5f"%(loss.item(),output_loss.item(),gradient_loss.item()))
                print("")

                save_image_batch(model, rgb_fixed, depth_fixed, predicted_dirs, num)
                num = num + 1
            if ((i+1) % 3000 == 0):
                output = outputs.cpu().detach().numpy()
                save_image_tensor(output,result_dirs,'output_depth_%d.png'%(model_num+1))
                save_image_tensor(origin,result_gt_dirs,'origin_depth_%d.png'%(model_num+1))
                torch.save(model.state_dict(), save_dir+'/epoch_%d_AE_depth_loss_%.4f.pkl' %(model_num+1,loss))
                model_num = model_num + 1
        if logger is not None:
            #######################################################################################################
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(losses.avg[0]))
            ################################ evalutating on validation set ########################################
            logger.reset_valid_bar()
            errors, error_names = validate(args, val_loader, model, epoch, logger,args.mode)
            
            ################# training log ############################
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
            logger.valid_writer.write(' * Avg {}'.format(error_string))

            for error, name in zip(errors, error_names):
                train_writer.add_scalar(name, error, epoch)

            # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
            decisive_error = errors[1]
            if best_error < 0:
                best_error = decisive_error
            # remember lowest error and save checkpoint
            is_best = decisive_error < best_error
            best_error = min(best_error, decisive_error)
            if is_best:
                torch.save(model, args.save_path/'AE_DtoD_model_best.pth.tar')
            
            with open(args.save_path/args.log_summary, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([loss, decisive_error])
        ##################################################################################
        ############### Saving output image and model ####################################
        if  (epoch % 1 == 0):
            print('\n','epoch: ',epoch+1,'  loss: ',loss.item())

            output = outputs.cpu().detach().numpy()
            save_image_tensor(output,result_dirs,'output_depth_%d.png'%(model_num+1))
            save_image_tensor(origin,result_gt_dirs,'origin_depth_%d.png'%(model_num+1))
            torch.save(model.state_dict(), save_dir+'/epoch_%d_AE_depth_loss_%.4f.pkl' %(model_num+1,loss))
            model_num = model_num + 1

        #####################################################################################
        ################### Extracting feature_map ##########################################
        if ((epoch+1) % 5 ==0 or epoch ==0):
            result_dir = result_dirs + '/epoch_%d_depth'%(epoch+1)      
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            with torch.no_grad():
                rft1, rft2, rft3, rft4, rft5, rft6, rft7, rout = model(inputs,istrain=True)
                ft1_gt, ft2_gt, ft3_gt, ft4_gt, ft5_gt, ft6_gt, ft7_gt, _ = DtoD_model(depths, istrain=True)
                dft1, dft2, dft3, dft4, dft5, dft6, dft7, _ = DtoD_model(rout, istrain=True)
                rftmap_list = [rft1, rft2, rft3, rft4, rft5, rft6, rft7]
                gt_ftmap_list = [ft1_gt, ft2_gt, ft3_gt, ft4_gt, ft5_gt, ft6_gt, ft7_gt]
                dftmap_list = [dft1, dft2, dft3, dft4, dft5, dft6, dft7]
            
            result_dir = result_dirs + '/epoch_%d_depth'%(epoch+1)          
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            for kk in range(len(rftmap_list)):
                ftmap_extract(args,num_sample_list[kk], figsize_x_list[kk], figsize_y_list[kk], d_range_list[kk], rftmap_list[kk],ftmap_height_list[kk], ftmap_width_list[kk], result_dir+'/RtoD', epoch,kk+1)
                ftmap_extract(args,num_sample_list[kk], figsize_x_list[kk], figsize_y_list[kk], d_range_list[kk], gt_ftmap_list[kk],ftmap_height_list[kk], ftmap_width_list[kk], result_dir+'/DtoD_gt', epoch,kk+1)
                ftmap_extract(args,num_sample_list[kk], figsize_x_list[kk], figsize_y_list[kk], d_range_list[kk], dftmap_list[kk],ftmap_height_list[kk], ftmap_width_list[kk], result_dir+'/DtoD', epoch,kk+1)
            
            print("featmap save is finished")
            
            save_image_tensor(origin,result_dir,'origin_depth.png')
            print("origin_depth save is finished")
            #####################################################################################
            #####################################################################################
    if logger is not None:
        logger.epoch_bar.finish()
    return loss

def train_AE_RtoD(args,model,DtoD_model,criterion_L2, criterion_L1, optimizer, dataset_loader,val_loader, batch_size, n_epochs,lr,logger,train_writer):
    global n_iter, best_error
    print("Training for %d epochs..." % n_epochs)
    num = 0
    model_num = 0
    data_iter = iter(dataset_loader)
    depth_fixed, rgb_fixed, _ = next(data_iter)
    depth_fixed = depth_fixed.cuda()
    rgb_fixed = rgb_fixed.cuda()

    predicted_dirs = './' + args.dataset + '_AE_RtoD_predicted_lr000%d_color_uNet_gen2_nogradf'%(lr*100000)
    result_dirs = './' + args.dataset + '_AE_RtoD_feat_result_lr000%d_color_uNet_gen2_nogradf/out'%(lr*100000)
    result_gt_dirs = './' + args.dataset + '_AE_RtoD_feat_result_lr000%d_color_uNet_gen2_nogradf/gt'%(lr*100000)  
    save_dir = './' + args.dataset + '_AE_RtoD_trained_model_lr000%d_color_uNet_gen2_nogradf'%(lr*100000)
    if((args.local_rank + 1)%4 == 0):
        if not os.path.exists(predicted_dirs):
            os.makedirs(predicted_dirs)
        if not os.path.exists(result_dirs):
            os.makedirs(result_dirs)
        if not os.path.exists(result_gt_dirs):
            os.makedirs(result_gt_dirs)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    H = depth_fixed.shape[2]
    W = depth_fixed.shape[3]
    num_sample_list = [16,64,64,64,64,64,16]
    figsize_x_list = [14,16,10,10,10,16,14]
    figsize_y_list = [7,8,5,5,5,8,7]

    if args.dataset != 'NYU':
        d_range_list = [4,2,2,4,2,2,4]
        ftmap_height_list = [H, int(H/2), int(H/8), int(H/16), int(H/8), int(H/2), H]
        ftmap_width_list = [W, int(W/2), int(W/8), int(W/16), int(W/8), int(W/2), W]
    else:
        d_range_list = [4,2,4,2,4,2,4]
        ftmap_height_list = [H, int(H/2), int(H/4), int(H/8), int(H/16), int(H/2), H]
        ftmap_width_list = [W, int(W/2), int(W/4), int(W/8), int(W/16), int(W/2), W]

    test_loss_dir = Path(args.save_path)
    test_loss_dir_rmse = str(test_loss_dir/'test_rmse_list.txt')
    test_loss_dir = str(test_loss_dir/'test_loss_list.txt')
    train_loss_dir = Path(args.save_path)
    train_loss_dir_rmse = str(train_loss_dir/'train_rmse_list.txt')
    train_loss_dir = str(train_loss_dir/'train_loss_list.txt')
    
    loss_list = []
    rmse_list = []
    train_loss_list = []
    train_rmse_list = []
    num_cnt = 0
    train_loss_cnt = 0
    if args.dataset == "KITTI":
        y1,y2 = int(0.40810811 * depth_fixed.size(2)), int(0.99189189 * depth_fixed.size(2))
        x1,x2 = int(0.03594771 * depth_fixed.size(3)), int(0.96405229 * depth_fixed.size(3))    ### Crop used by Garg ECCV 2016
        '''
        y1,y2 = int(0.3324324 * depth_fixed.size(2)), int(0.91351351 * depth_fixed.size(2))
        x1,x2 = int(0.0359477 * depth_fixed.size(3)), int(0.96405229 * depth_fixed.size(3))     ### Crop used by Godard CVPR 2017   
        '''
        print(" - valid y range: %d ~ %d"%(y1,y2))
        print(" - valid x range: %d ~ %d"%(x1,x2))
    for epoch in tqdm(range(n_epochs)):
        if args.dataset == "KITTI":
            crop_mask = depth_fixed != depth_fixed
            #print('crop_mask size: ',crop_mask.size())
            crop_mask[:,:,y1:y2,x1:x2] = 1
        if logger is not None:
            logger.epoch_bar.update(epoch)
            ####################################### one epoch training #############################################
            logger.reset_train_bar()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(precision=4)
        ################ train mode ####################
        model.train()
        ################################################
        end = time.time()
        if logger is not None:
            logger.train_bar.update(0)
        for i,(gt_data, rgb_data, gt_data_2) in enumerate(dataset_loader):
            # data loading time
            if logger is not None:
                data_time.update(time.time() - end)
            # get the inputs
            inputs = rgb_data
            depths = gt_data
            if args.dataset != "KITTI":
                gt_data_2 = None
            # If gt_data_2 is None ==> NYU dataset!
            if gt_data_2 is not None:
                sparse_depths = gt_data_2.cuda()
                sparse_depths = Variable(sparse_depths)

            origin = depths
            inputs = inputs.cuda()
            depths = depths.cuda()

            # wrap them in Variable
            inputs, depths = Variable(inputs), Variable(depths)
            
            ########################################
            ### Train the AutoEncoder (Generator) ###
            ########################################
            
            '''AutoEncoder loss'''
            outputs = model(inputs,istrain=False)
            
            if args.mode != 'RtoD_single':
                with torch.no_grad():
                    ft_map1_tar, ft_map2_tar, ft_map3_tar, ft_map4_tar, _, _, _, _ = DtoD_model(depths, istrain=True)
            if args.mode != 'RtoD_single':
                with torch.no_grad():
                    ft_map1, ft_map2, ft_map3, ft_map4, _, _, _, _ = DtoD_model(outputs, istrain=True)
            # masking valied area
            if gt_data_2 is not None:
                valid_mask = sparse_depths > -1
                valid_mask = valid_mask[:,0,:,:].unsqueeze(1)
                if(crop_mask.size(0) != valid_mask.size(0)):
                    crop_mask = crop_mask[0:valid_mask.size(0),:,:,:]

            diff = outputs - depths
            diff_abs = torch.abs(diff)
            diff_2 = torch.pow(outputs-depths,2)
            c = 0.2*torch.max(diff_abs.detach())         
            mask2 = torch.gt(diff_abs.detach(),c)
            diff_abs[mask2] = (diff_2[mask2]+(c*c))/(2*c)
            if gt_data_2 is not None:
                diff_abs[~crop_mask] = 0.1*diff_abs[~crop_mask]      
                diff_abs[crop_mask&(~valid_mask)] = 0.3*diff_abs[crop_mask&(~valid_mask)]
            output_loss = 3*diff_abs.mean()
            
            diff2_clone = diff_2.clone().detach()
            rmse_loss = torch.sqrt(diff2_clone.mean())

            ################# BerHu Loss #########################
            latent_loss = torch.tensor(0.).cuda()
            if args.mode != 'RtoD_single':
                latent1 = criterion_L2(ft_map1, ft_map1_tar.detach())
                latent2 = 2.5*criterion_L2(ft_map2, ft_map2_tar.detach())
                latent3 = 14*criterion_L2(ft_map3, ft_map3_tar.detach())
                latent4 = 12*criterion_L2(ft_map4, ft_map4_tar.detach()) 
                #print("latent1 : ",latent1.item(),"latent2 : ",latent2.item(),"latent3 : ",latent3.item(),"latent4 : ",latent4.item())
                latent_loss = 1.5*((latent1 + latent2 + latent3 + latent4)/4) 
            ################# Latent Loss #########################
            #gradient_loss = imgrad_loss(outputs, depths) ## for kitti
            ##gradient_loss = 3.5* imgrad_loss(outputs, depths) ## for NYU
            ################# gradient loss #######################
            '''
            grad_latent_loss = torch.tensor(0.).cuda()
            if args.mode != 'RtoD_single':
                grad_latent1 = imgrad_loss(ft_map1, ft_map1_tar.detach())
                grad_latent2 = 1.5*imgrad_loss(ft_map2, ft_map2_tar.detach())
                grad_latent3 = 4*imgrad_loss(ft_map3, ft_map3_tar.detach()) ##for kitti
                ##grad_latent3 = 2*imgrad_loss(ft_map3, ft_map3_tar.detach()) ##for NYU
                grad_latent4 = 2.3*imgrad_loss(ft_map4, ft_map4_tar.detach())
                #print("g_latent1 : ",grad_latent1.item(),"g_latent2 : ",grad_latent2.item(),"g_latent3 : ",grad_latent3.item(),"g_latent4 : ",grad_latent4.item())
                ##grad_latent_loss = ((grad_latent1 + grad_latent2 + grad_latent3 + grad_latent4)/4.0) ## for kitti
                grad_latent_loss = ((grad_latent1 + grad_latent2 + grad_latent3 + grad_latent4)/4.0) ## for NYU
            ################# gradient latent loss ################
            grad_loss = (gradient_loss + grad_latent_loss)
            '''
            ################# gradient total loss ################
            depth_smoothness_tot = 0.1*depth_smoothness(outputs,inputs) 
            depth_smoothness_loss = torch.mean(torch.abs(depth_smoothness_tot))
            ################# smoothness loss ######################
            #loss = output_loss + latent_loss + grad_loss + depth_smoothness_loss
            loss = output_loss + latent_loss + depth_smoothness_loss  
            if logger is not None:
                if i > 0 and n_iter % args.print_freq == 0:
                    train_writer.add_scalar('output_loss', output_loss.item(), n_iter)
                    train_writer.add_scalar('latent_loss', latent_loss.item(), n_iter)
                    train_writer.add_scalar('total_loss', loss.item(), n_iter)
                # record loss and EPE
                losses.update(loss.item(), args.batch_size)
            # zero the parameter gradients and backward & ptimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            if logger is not None:
                batch_time.update(time.time() - end)
                end = time.time()

                with open(args.save_path/args.log_full, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow([loss.item(), output_loss.item(), latent_loss.item()])
                logger.train_bar.update(i+1)
                if i % args.print_freq == 0:
                    logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
                n_iter += 1
            if i >= args.epoch_size - 1:
                break
            ### KITTI's learning decay ###
            if (epoch>2):
                if ((i+1) % 2200 == 0):
                    if (lr < 0.00002):
                        lr -= (lr / 100)
                    else :
                        lr -= (lr / 60)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decayed learning rates, lr: {}'.format(lr))
            ### NYU's learning decay ###
            '''
            if (epoch>6):
                if ((i+1) % 1900 == 0):
                    if (lr < 0.00002):
                        lr -= (lr / 200)
                    else :
                        lr -= (lr / 40)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decayed learning rates, lr: {}'.format(lr))
            '''
            if ((i+1) % 100 == 0):
                if((args.local_rank + 1)%4 == 0):
                    print("epoch: %d,  %d/%d"%(epoch+1,i+1,args.epoch_size))
                    if args.mode == 'RtoD':
                        print("total_loss: %5f, output_loss: %5f, smoothness_loss: %5f, latent_loss: %5f"%(loss.item(),output_loss.item(),
                            depth_smoothness_loss.item(),latent_loss.item()))
                        #print("grad_loss: %5f, gradient_loss: %5f, grad_latent_loss: %5f"%(grad_loss.item(), gradient_loss.item(), grad_latent_loss.item()))
                    elif args.mode == 'RtoD_single':
                        print("total_loss: %5f, output_loss: %5f, smoothness_loss: %5f"%(loss.item(),output_loss.item(),
                        depth_smoothness_loss.item()))
                        print("grad_loss: %5f"%(grad_loss.item()))
                '''
                total_loss = loss.item()
                rmse_loss = rmse_loss.item()
                loss_pdf = "train_loss.pdf"
                rmse_pdf = "train_rmse.pdf"
                train_loss_cnt = train_loss_cnt + 1
                all_plot(args.save_path,total_loss, rmse_loss, train_loss_list, train_rmse_list, train_loss_dir,train_loss_dir_rmse,loss_pdf, rmse_pdf, train_loss_cnt,True)
                print("")
                '''
                if ((i+1) % 700 == 0):
                    save_image_batch(model,rgb_fixed, depth_fixed, predicted_dirs, num)
                    num = num + 1
            if ((i+1) % 700 == 0):
                '''
                test_loss, rmse_test_loss = validate_in_test(args, val_loader, model,DtoD_model,n_epochs, logger,args.mode, crop_mask,criterion_L2)
                loss_pdf = "test_loss.pdf"
                rmse_pdf = "test_rmse.pdf"
                num_cnt = num_cnt + 1
                if((args.local_rank + 1)%4 == 0):
                    print('%d th test_set_loss :  %.4f'%(num_cnt,test_loss))
                all_plot(args.save_path,test_loss, rmse_test_loss, loss_list, rmse_list, test_loss_dir,test_loss_dir_rmse,loss_pdf, rmse_pdf, num_cnt,False)             
                '''
                if((args.local_rank + 1)%4 == 0):
                    output = outputs.cpu().detach().numpy()
                    save_image_tensor(output,result_dirs,'output_depth_%d.png'%(model_num+1))
                    save_image_tensor(origin,result_gt_dirs,'origin_depth_%d.png'%(model_num+1))

                    torch.save(model.state_dict(), save_dir+'/epoch_%d_AE_depth_loss_%.4f.pkl' %(model_num+1,loss))
                model_num = model_num + 1
        if logger is not None:
            ######################################################################################################
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(losses.avg[0]))
            ################################ evalutating on validation set ########################################
            logger.reset_valid_bar()
            errors, error_names = validate(args, val_loader, model, epoch, logger,args.mode)
            
            ################# training log ############################
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
            logger.valid_writer.write(' * Avg {}'.format(error_string))

            for error, name in zip(errors, error_names):
                train_writer.add_scalar(name, error, epoch)

            # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
            decisive_error = errors[1]
            if best_error < 0:
                best_error = decisive_error

            # remember lowest error and save checkpoint
            is_best = decisive_error < best_error
            best_error = min(best_error, decisive_error)
            if is_best:
                torch.save(model, args.save_path/'AE_RtoD_model_best.pth.tar')
            
            with open(args.save_path/args.log_summary, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([loss, decisive_error])
            ###########################################################
        ##if ((epoch+1) % 2) and ((epoch+1) > n_epochs/2):
        #if (((epoch+1) % 2) and epoch>5):
        '''
        if epoch % 1 == 0:
            #print('\n','epoch: ',epoch+1,'  loss: ',loss.item())
            print('output_loss: ',output_loss.item(),'  latent_loss: ',latent_loss.item())
            
            output = outputs.cpu().detach().numpy()
            save_image_tensor(output,result_dirs,'output_depth_%d.png'%(model_num+1))
            save_image_tensor(origin,result_gt_dirs,'origin_depth_%d.png'%(model_num+1))
            
            torch.save(model.state_dict(), save_dir+'/epoch_%d_AE_depth_loss_%.4f.pkl' %(model_num+1,loss))
            model_num = model_num + 1
        '''
        #####################################################################################
        ################### Extracting feature_map ##########################################
        if ((epoch+1) % 10 ==0 or epoch ==0):
            if((args.local_rank + 1)%4 == 0):    
                with torch.no_grad():
                    rft1, rft2, rft3, rft4, rft5, rft6, rft7, rout = model(inputs,istrain=True)
                    ft1_gt, ft2_gt, ft3_gt, ft4_gt, ft5_gt, ft6_gt, ft7_gt, _ = DtoD_model(depths, istrain=True)
                    dft1, dft2, dft3, dft4, dft5, dft6, dft7, _ = DtoD_model(rout, istrain=True)
                    rftmap_list = [rft1, rft2, rft3, rft4, rft5, rft6, rft7]
                    gt_ftmap_list = [ft1_gt, ft2_gt, ft3_gt, ft4_gt, ft5_gt, ft6_gt, ft7_gt]
                    dftmap_list = [dft1, dft2, dft3, dft4, dft5, dft6, dft7]
                
                result_dir = result_dirs + '/epoch_%d_depth'%(epoch+1)          
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                
                for kk in range(len(rftmap_list)):
                    ftmap_extract(args,num_sample_list[kk], figsize_x_list[kk], figsize_y_list[kk], d_range_list[kk], rftmap_list[kk],ftmap_height_list[kk], ftmap_width_list[kk], result_dir+'/RtoD', epoch,kk+1)
                    ftmap_extract(args,num_sample_list[kk], figsize_x_list[kk], figsize_y_list[kk], d_range_list[kk], gt_ftmap_list[kk],ftmap_height_list[kk], ftmap_width_list[kk], result_dir+'/DtoD_gt', epoch,kk+1)
                    ftmap_extract(args,num_sample_list[kk], figsize_x_list[kk], figsize_y_list[kk], d_range_list[kk], dftmap_list[kk],ftmap_height_list[kk], ftmap_width_list[kk], result_dir+'/DtoD', epoch,kk+1)
                
                print("featmap save is finished")
                
                inputs_ = inputs.cpu().detach().numpy()
                save_image_tensor(origin,result_dir,'origin_depth.png')
                save_image_tensor(inputs_,result_dir,'origin_input.png')
                
                print("origin_depth save is finished")
                print("origin_image save is finished")
            #####################################################################################
            #####################################################################################
    if logger is not None:
        logger.epoch_bar.finish()
    
    return loss, output_loss, latent_loss


if __name__ == "__main__":
    main()