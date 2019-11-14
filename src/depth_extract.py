from AE_model_unet import *
import os
import torch.backends.cudnn as cudnn
import time
from path import Path
from imageio import imread
from PIL import Image
import scipy.misc
from torch.autograd import Variable
import collections
import argparse

parser = argparse.ArgumentParser(description='Pretrained Depth AutoEncoder',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_dir', type=str, default = "../model/GDN_RtoD_pretrained.pkl")                                 
parser.add_argument('--gpu_num', type=str, default = "0")
args = parser.parse_args()

def load_as_float(path):
    return imread(path).astype(np.float32)

class Resize(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    'nearest' or 'bilinear'
    """
    def __init__(self, interpolation='bilinear'):
        self.interpolation = interpolation
    def __call__(self, img,size, img_type = 'rgb'):
        assert isinstance(size, int) or isinstance(size, float) or \
               (isinstance(size, collections.Iterable) and len(size) == 2)
        if img_type == 'rgb':
            if img.ndim == 3:
                return scipy.misc.imresize(img, size, self.interpolation)
            elif img.ndim == 2:
                img = scipy.misc.imresize(img, size, self.interpolation)
                img_tmp = np.zeros((img.shape[0], img.shape[1],1),dtype=np.float32)
                img_tmp[:,:,0] = img[:,:]
                img = img_tmp
                return img
        elif img_type == 'depth':
            if img.ndim == 2:
                img = scipy.misc.imresize(img, size, self.interpolation, 'F')
            elif img.ndim == 3:
                img = scipy.misc.imresize(img[:,:,0], size, self.interpolation, 'F')
            img_tmp = np.zeros((img.shape[0], img.shape[1],1),dtype=np.float32)
            img_tmp[:,:,0] = img[:,:]
            img = img_tmp
            return img
        else:
            RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
upsampling = nn.functional.interpolate
resize = Resize()

ae = AutoEncoder()
ae = ae.cuda()
ae = nn.DataParallel(ae)
ae.load_state_dict(torch.load(args.model_dir))
ae = ae.eval()

cudnn.benchmark = True
torch.cuda.synchronize()
sum = 0.0

scene = Path("../example/demo_input")
png_list = (scene.files("*.png"))
jpg_list = (scene.files("*.jpg"))
img_list = sorted(png_list + jpg_list)
sample = []
filenames = []
org_H_list = []
org_W_list = []
for filename in img_list:
    img = load_as_float(filename)
    org_H_list.append(img.shape[0])
    org_W_list.append(img.shape[1])
    img = resize(img,(128,416),'rgb')
    img = img.transpose(2,0,1)
    img = torch.tensor(img,dtype=torch.float32)
    img = img.unsqueeze(0)
    img = img/255
    img = (img-0.5)/0.5
    #img = upsampling(img, (128,416),mode='bilinear', align_corners = False)
    img = Variable(img)
    sample.append(img)
    filenames.append(filename.split('/')[-1])

print("sample len: ",len(sample))
'''
for i in range(100):
     torch.cuda.synchronize()
     start = time.time()
     result = ae(img,istrain=False)
     tmp = time.time() - start
     print(tmp)
     sum += tmp
     print("----",i," th iter")
'''
i=0
result_dir = "../example/demo_output/"
k=0
t=0
img_ = None
for tens in sample:
    filename = filenames[i]
    org_H = org_H_list[i]
    org_W = org_W_list[i]
    torch.cuda.synchronize()

    start = time.time()
    #print(tens.size())
    img = ae(tens,istrain=False)
    tmp = time.time() - start
    #print(tmp)
    if i>0:
        sum += tmp
    #print(img.size())
    img = upsampling(img, (128,416),mode='bilinear', align_corners = False)
    img = img[0].cpu().detach().numpy()
    #print(img.shape)
    if img.shape[0] == 3:
        img_ = np.empty([128,416,3])
        img_[:,:,0] = img[0,:,:]
        img_[:,:,1] = img[1,:,:]
        img_[:,:,2] = img[2,:,:]
    elif img.shape[0] == 1:
        img_ = np.empty([128,416])
        img_[:,:] = img[0,:,:]
    img_ = resize(img_, (org_H, org_W), 'rgb')
    if img_.shape[2] == 1:
        img_ = img_[:,:,0]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print(result_dir + filename)
    print(img_.shape)
    scipy.misc.imsave(result_dir + 'out_' + filename,img_)
    i = i + 1
    print("----",i," th iter")
    
print("Depth estimation demo is finished")
print("Avg time: ",sum/len(sample))
