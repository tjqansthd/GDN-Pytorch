from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from IPython import display
import itertools
import torch.nn as nn
import os
from torchvision.utils import save_image

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
        array = array.transpose(1,2,0)
    return array

def ftmap_extract(args, num_samples, figsize_x, figsize_y, d_range, ft_map,ft_map_height, ft_map_width, result_dir, epoch, ft_idx):
    if((args.local_rank + 1)%4 == 0):
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    num_test_samples = num_samples
    size_figure_grid = int(math.sqrt(num_test_samples))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(figsize_x, figsize_y))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for d in range(d_range):
        for k in range(num_test_samples):
            i = k//size_figure_grid
            j = k%size_figure_grid
            ax[i,j].cla()
            ax[i,j].imshow(ft_map[0][k+num_test_samples*d,:,:].data.cpu().numpy().reshape(ft_map_height, ft_map_width), cmap='Greys')
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.savefig(result_dir +'/ftmap' + str(ft_idx) + '_epoch_%d_%d.png'%(epoch+1,d))

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()

    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
#   grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    return grad_y, grad_x

def imgrad_loss(pred, gt):
    N,C,_,_ = pred.size()
    grad_y, grad_x = imgrad(pred)
    grad_y_gt, grad_x_gt = imgrad(gt)
    return (torch.mean(torch.abs(grad_y - grad_y_gt)) + torch.mean(torch.abs(grad_x - grad_x_gt)))

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)

    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def gradient_x(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx

def gradient_y(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = img.size()
    h = s[2]
    w = s[3]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(nn.functional.interpolate(img,
                           size=[nh, nw], mode='bilinear',
                           align_corners=True))
    return scaled_imgs

def depth_smoothness(depth, img):
    depth_gradients_x = gradient_x(depth)
    depth_gradients_y = gradient_y(depth)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1,keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1,keepdim=True))

    smoothness_x = depth_gradients_x * weights_x
    smoothness_y = depth_gradients_y * weights_y

    return (torch.abs(smoothness_x) + torch.abs(smoothness_y))

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image_tensor(tensor_img,img_dir,filename):
    input_ = tensor_img[0]
    if(tensor_img.shape[1]==1):
        input__ = np.empty([tensor_img.shape[2],tensor_img.shape[3]])
        input__[:,:] = input_[0,:,:]
    elif(tensor_img.shape[1]==3):
        input__ = np.empty([tensor_img.shape[2], tensor_img.shape[3],3])
        input__[:,:,0] = input_[0,:,:]
        input__[:,:,1] = input_[1,:,:]
        input__[:,:,2] = input_[2,:,:]
    else:
        print("file dimension is not proper!!")
        exit()
    scipy.misc.imsave(img_dir + '/' + filename,input__)

def save_image_batch(model,rgb_fixed, depth_fixed, predicted_dirs, num):
    B = depth_fixed.size(0)
    H = depth_fixed.size(2)
    W = depth_fixed.size(3)
    with torch.no_grad():
        img_list = [rgb_fixed]
        outputs_fixed = model(rgb_fixed,istrain=False)
        outputs_fixed_ = torch.empty([B,3,H,W]).cuda()
        depth_fixed_ = torch.empty([B,3,H,W]).cuda()
        for r in range(3):
            outputs_fixed_[:,r,:,:] = outputs_fixed[:,0,:,:]
            depth_fixed_[:,r,:,:] = depth_fixed[:,0,:,:]
        img_list.append(depth_fixed_) 
        img_list.append(outputs_fixed_)
        img_concat = torch.cat(img_list, dim=3)
        sample_path = os.path.join(predicted_dirs, 'predicted_depth_{}.jpg'.format(num))
        save_image(denorm(img_concat.data.cpu()), sample_path, nrow=1, padding=0)

def plot_loss(data, apath, epoch,train,filename):
    axis = np.linspace(1, epoch, epoch)
    
    label = 'Total Loss'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(data), label=label)
    plt.legend()
    if train is False:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('x100 = Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.savefig(os.path.join(apath, filename))
    plt.close(fig)

def all_plot(save_dir,tot_loss, rmse, loss_list, rmse_list, tot_loss_dir,rmse_dir,loss_pdf, rmse_pdf, count,istrain):
    open_type = 'a' if os.path.exists(tot_loss_dir) else 'w'
    loss_log_file = open(tot_loss_dir, open_type)
    rmse_log_file = open(rmse_dir,open_type)

    loss_list.append(tot_loss)
    rmse_list.append(rmse)

    plot_loss(loss_list, save_dir, count, istrain, loss_pdf)
    plot_loss(rmse_list, save_dir, count, istrain, rmse_pdf)
    loss_log_file.write(('%.5f'%tot_loss) + '\n')
    rmse_log_file.write(('%.5f'%rmse) + '\n')
    loss_log_file.close()
    rmse_log_file.close()