from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize
import scipy
import scipy.ndimage
import numbers
import collections

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images

class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img

class Merge(object):
    """Merge a group of images
    """
    def __init__(self, axis=-1):
        self.axis = axis
    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray)
                        for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes]
                       ), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, images):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            if(im.ndim==2):
                im_tmp = np.zeros((self.height,self.width,1),dtype=np.float32)
                im_tmp[:,:,0] = im[:,:]
                im = im_tmp
            im = im.transpose((2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors

class RandomCropNumpy(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, random_state=np.random):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.random_state = random_state

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        if w == tw and h == th:
            return img
        elif h == th:
            y1 = 0
            x1 = self.random_state.randint(0, w - tw)
        elif w == tw:
            y1 = self.random_state.randint(0, h - th)
            x1 = 0
        else:
            y1 = self.random_state.randint(0, h - th)
            x1 = self.random_state.randint(0, w - tw)

        return img[y1:y1 + th, x1: x1 + tw, :]

class RandomColor(object):
    """Multiply numpy.ndarray (H x W x C) globally
    """

    def __init__(self, multiplier_range=(0.8, 1.2), random_state=np.random):
        assert isinstance(multiplier_range, tuple)
        self.multiplier_range = multiplier_range
        self.random_state = random_state

    def __call__(self, image):
        mult = self.random_state.uniform(self.multiplier_range[0],
                                         self.multiplier_range[1])
        if isinstance(image, np.ndarray):
            return np.clip(image * mult, 0, 255)
        else:
            raise Exception('unsupported type')

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images):
        if random.random() < 0.5:
            output_images = [np.copy(np.fliplr(im)) for im in images]
        else:
            output_images = images
        return output_images

class RandomAffineZoom(object):
    def __init__(self, scale_range=(1.0, 1.5), random_state=np.random):
        assert isinstance(scale_range, tuple)
        self.scale_range = scale_range
        self.random_state = random_state

    def __call__(self, image):
        scale = self.random_state.uniform(self.scale_range[0],
                                          self.scale_range[1])
        if isinstance(image, np.ndarray):
            af = AffineTransform(scale=(scale, scale))
            image = warp(image, af.inverse)
            rgb = image[:, :, 0:3]
            depth = image[:, :, 3:4] / scale
            return np.concatenate([rgb, depth], axis=2)
        else:
            raise Exception('unsupported type')

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images):
        #print("images[1].shape: ",images[1].shape)
        in_h, in_w, _ = images[1].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]
        images[1]
        return cropped_images

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
            return scipy.misc.imresize(img, size, self.interpolation)
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
class CenterCrop(object):
    """Crops the given ``numpy.ndarray`` at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for center crop.
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (H x W x C)): Image to be cropped.
        Returns:
            img (numpy.ndarray (H x W x C)): Cropped image.
        """
        i, j, h, w = self.get_params(img[0], self.size)

        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not(_is_numpy_image(img[0])):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img[1].ndim == 3:
            return [im[i:i+h, j:j+w, :] for im in img]
        elif img[1].ndim == 2:
            return [im[i:i+h, j:j+w] for im in img]
        else:
            raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

class RandomRotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self, angle_range=(0.0, 360.0), axes=(0, 1), mode='reflect', random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode

    def __call__(self, image):
        angle = self.random_state.uniform(
            self.angle_range[0], self.angle_range[1])
        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(
                image, angle, reshape=False, axes=self.axes, mode=self.mode)
            return np.clip(image, mi, ma)
        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        else:
            raise Exception('unsupported type')

class Split(object):
    """Split images into individual arraies
    """

    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
        assert all([isinstance(s, slice) for s in slices_]
                   ), 'slices must be consist of slice instances'
        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                sl = [slice(None)] * image.ndim
                sl[self.axis] = s
                ret.append(image[tuple(sl)])
            return ret
        else:
            raise Exception("obj is not an numpy array")