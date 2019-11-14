from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np

#@torch.no_grad()
def compute_errors(gt_np, gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3,rmse_tot,rmse_log_tot = 0,0,0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = pred[0] != pred[0]
        crop_mask = crop_mask[0,:,:]
        #print('crop_mask size: ',crop_mask.size())
        '''
        y1,y2 = int(0.40810811 * pred.size(2)), int(0.99189189 * pred.size(2))
        x1,x2 = int(0.03594771 * pred.size(3)), int(0.96405229 * pred.size(3))    ### Crop used by Garg ECCV 2016
        
        '''
        y1,y2 = int(0.3324324 * pred.size(2)), int(0.91351351 * pred.size(2))
        x1,x2 = int(0.0359477 * pred.size(3)), int(0.96405229 * pred.size(3))     ### Crop used by Godard CVPR 2017
        
        crop_mask[y1:y2,x1:x2] = 1
    for current_gt_np, current_gt, current_pred in zip(gt_np, gt, pred):

        current_gt = current_gt[0,:,:]
        current_gt_np = current_gt_np[0,:,:]
        current_pred = current_pred[0,:,:]
        
        current_pred = ((current_pred-current_pred.min())/(current_pred.max()-current_pred.min()))
        current_gt = ((current_gt-current_gt.min())/(current_gt.max()-current_gt.min()))
        #current_gt_np = ((current_gt_np-current_gt_np.min())/(current_gt_np.max()-current_gt_np.min()))
        current_gt_np = ((current_gt_np+1.0)/2.0)


        current_gt = current_gt*80
        current_pred = current_pred*80
        current_gt_np = current_gt_np*80

        '''
        current_gt = ((current_gt-current_gt.min())/(current_gt.max()-current_gt.min()))
        current_pred = ((current_pred-current_pred.min())/(current_pred.max()-current_pred.min()))
        current_gt_np = ((current_gt_np-current_gt_np.min())/(current_gt_np.max()-current_gt_np.min()))
        
        #valid = (current_gt_np > 1e-3)&(current_gt >1e-3)
        current_gt = current_gt*80
        current_pred = current_pred*80
        current_gt_np = current_gt_np*80
        '''
        '''
        current_pred = np.array(current_pred.cpu().detach())
        current_pred = current_pred.transpose((1, 2, 0))
        current_pred = cv2.resize(current_pred,(1216,352),interpolation=cv2.INTER_LINEAR)
        current_pred = current_pred.transpose((2, 0, 1))
        current_pred = torch.tensor(current_pred)
        current_pred = current_pred.cuda()

        current_gt = np.array(current_gt.cpu())
        current_gt = current_gt.transpose((1,2,0))
        current_gt = cv2.resize(current_gt,(1216,352),interpolation=cv2.INTER_LINEAR)
        current_gt = current_gt.transpose((2,0,1))
        current_gt = torch.tensor(current_gt)
        current_gt = current_gt.cuda()
        
        crop_mask = current_gt != current_gt
        y1,y2 = int(0.40810811 * current_gt.size(0)), int(0.99189189 * current_gt.size(0))
        x1,x2 = int(0.03594771 * current_gt.size(1)), int(0.96405229 * current_gt.size(1))
        crop_mask[y1:y2,x1:x2] = 1
        '''
        valid = (current_gt_np < 80)&(current_gt < 80) &(current_gt_np > 1) & (current_gt > 1)
        #valid = (current_gt_np < 50)&(current_gt_np>0)&(current_gt>0)
        #valid = valid & (current_gt_np < 80)&(current_gt<80)
        if crop:
            valid = valid & crop_mask
        #valid_gt = current_gt_np[valid].clamp(1, 80)
        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]
        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)
        valid_pred = valid_pred.clamp(1,80)
        ##print("torch.median(valid_gt)/torch.median(valid_pred): ",(torch.median(valid_gt)/torch.median(valid_pred)).item())

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()
        rmse = (valid_gt - valid_pred) ** 2
        rmse_tot += torch.sqrt(torch.mean(rmse))
        rmse_log = (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
        rmse_log_tot += torch.sqrt(torch.mean(rmse_log))
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3,rmse_tot,rmse_log_tot]]

def compute_errors_NYU(gt, pred, crop=True):
    abs_diff, abs_rel, log10, a1, a2, a3,rmse_tot,rmse_log_tot = 0,0,0,0,0,0,0,0
    batch_size = gt.size(0)
    if crop:
        crop_mask = pred[0] != pred[0]
        crop_mask = crop_mask[0,:,:]
        y1,y2 = int(0.0359477 * pred.size(2)), int(0.96405229 * pred.size(2))
        x1,x2 = int(0.0359477 * pred.size(3)), int(0.96405229 * pred.size(3))     ### Crop used by Godard CVPR 2017
        
        crop_mask[y1:y2,x1:x2] = 1
    for current_gt, current_pred in zip(gt, pred):

        current_gt = current_gt[0,:,:]
        current_pred = current_pred[0,:,:]
        
        current_pred = ((current_pred-current_pred.min())/(current_pred.max()-current_pred.min()))
        current_gt = ((current_gt-current_gt.min())/(current_gt.max()-current_gt.min()))
        #current_pred = ((current_pred+1.0)/2.0)
        #current_gt = ((current_gt+1.0)/2.0)
        current_gt = current_gt*10
        current_pred = current_pred*10

        valid = (current_gt<10)&(current_gt>0)
        if crop:
            valid = valid & crop_mask
        #valid_gt = current_gt_np[valid].clamp(1e-3, 10)
        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]
        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)
        valid_pred = valid_pred.clamp(1e-3,10)
        ##print("torch.median(valid_gt)/torch.median(valid_pred): ",(torch.median(valid_gt)/torch.median(valid_pred)).item())

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()
        rmse = (valid_gt - valid_pred) ** 2
        rmse_tot += torch.sqrt(torch.mean(rmse))
        rmse_log = (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
        rmse_log_tot += torch.sqrt(torch.mean(rmse_log))
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        log10 += torch.mean(torch.abs(torch.log10(valid_gt)-torch.log10(valid_pred)))

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, log10, a1, a2, a3,rmse_tot,rmse_log_tot]]

def compute_errors_Make3D(gt_np, gt, pred):
    abs_diff, abs_rel, ave_log10,rmse_tot = 0,0,0,0
    batch_size = gt.size(0)

    for current_gt_np, current_gt, current_pred in zip(gt_np, gt, pred):

        current_gt = current_gt[0,:,:].unsqueeze(0)
        current_gt_np = current_gt_np[0,:,:].unsqueeze(0)

        
        current_gt = ((current_gt-current_gt.min())/(current_gt.max()-current_gt.min()))
        current_pred = ((current_pred-current_pred.min())/(current_pred.max()-current_pred.min()))
        current_gt_np = ((current_gt_np-current_gt_np.min())/(current_gt_np.max()-current_gt_np.min()))
        valid = (current_gt_np > 1e-2)&(current_gt > 1e-2)
        current_gt = current_gt*80
        current_pred = current_pred*80
        current_gt_np = current_gt_np*80
        
        valid = valid & (current_gt_np < 80)&(current_gt<80)

        valid_gt = current_gt[valid].clamp(1e-2, 80)
        valid_pred = current_pred[valid].clamp(1e-2, 80)
        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)
        ##print("torch.median(valid_gt)/torch.median(valid_pred): ",(torch.median(valid_gt)/torch.median(valid_pred)).item())
        rmse = (valid_gt - valid_pred) ** 2
        rmse_tot += torch.sqrt(torch.mean(rmse))
        ave_log10 += torch.mean(torch.abs(torch.log10(valid_gt)-torch.log10(valid_pred)))
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, ave_log10,rmse_tot]]