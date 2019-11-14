import torch.utils.data as data
from PIL import Image
import numpy as np
from imageio import imread
from path import Path
import random
import torch
import time
import cv2
from PIL import ImageFile
from transform_list import Resize
from scipy.misc import imresize
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_as_float(path):
    return imread(path).astype(np.float32)

def load_as_float_2(path):
    return np.array(Image.open(path),dtype=np.float32)

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    if(np.max(depth_png)<=255):
        print("max_value: ",np.max(depth_png))
        print("file_name: ",filename)
        assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / 256.
    #depth[depth_png == 0] = -1.
    return depth

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root,args, seed=None, train=True, transform=None, target_transform=None, mode = "DtoD"):
        np.random.seed(int(time.time()))                                                                    ## 랜덤하게 셔플하기 위한 장치
        random.seed(time.time())                                                                       ## 랜덤하게 셔플하기 위한 장치
        self.root = Path(root)
        if args.img_test is False:                                                                  ## Dataset의 경로 path설정
            scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'               ## Dataset의 폴더 리스트가 담긴 txt파일 경로
        elif args.img_test is True:
            scene_list_path = self.root/'test_scenes_2.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]               ## txt파일 내의 모든 폴더 리스트가 배열로써 self.scenes에 담긴다
        self.transform = transform                                                              ## 매개변수로 받은 transform 클래스 대입
        self.train = train
        self.mode = mode
        self.args = args
        self.crawl_folders()                                                                    ## SequenceFolder 클래스의 crawl_folders 실행

    def crawl_folders(self):                                                   
        sequence_set = []                                                                       ## 처음 sequence_set을 빈 list로 초기화
        for scene in self.scenes:
            depth_scene = scene/'color_gt2'                                                          ## g.t depth파일들이 담긴 경로
            sparse_depth_scene = scene/'gt'
            imgs = sorted(scene.files('*.jpg'))                                                 ## input 이미지들을 정렬
            gt_depth_np = None
            gt_depth = sorted(depth_scene.files('*.png'))
            #gt_depth_np = sorted(scene.files('*.npy'))
            gt_depth_np = sorted(sparse_depth_scene.files('*.png'))
            for i in range(len(imgs)):
                sample = {'gt': gt_depth[i], 'rgb': imgs[i], 'gt_np': gt_depth_np[i]}
                sequence_set.append(sample)
        if (self.args.img_test is False) or (self.train is True):
            random.shuffle(sequence_set)                                                            ## 모두 추가된 sequence_set를 셔플
        self.samples = sequence_set                                                             ## self.samples에 셔플이 완료된 sequence_set 담기

    def __getitem__(self, index):
        sample = self.samples[index]                                                            ## 주어지는 index에 해당하는 samples안의 이미지,gt pair data 추출
        gt_img = load_as_float(sample['gt'])                                                    ## sample에서 'gt이미지'에 해당하는 경로의 이미지 파일을 로드                                   
        rgb_img = load_as_float(sample['rgb'])
        ##gt_img_sparse = depth_read(sample['gt_np'])
        ##gt_img_sparse = cv2.resize(gt_img_sparse,(416,128),interpolation=cv2.INTER_LINEAR)    ## gt가 official kitti gt일때
        gt_img_sparse = load_as_float(sample['gt_np'])                                          ## gt가 generate된 gt일때
        #print("gt_img size: ",gt_img.shape)
        #print("rgb_img size: ",rgb_img.shape)
        #print("gt_img_sparse size: ",gt_img_sparse.shape)
        if self.transform is not None:
            imgs = self.transform([gt_img] + [rgb_img] + [gt_img_sparse])
            gt_img = imgs[0]
            rgb_img = imgs[1]
            gt_img_sparse = imgs[2]
        '''
        gt_img_np = np.load(sample['gt_np']).astype(np.float32)
        im_tmp = np.zeros((128,416,3),dtype=np.float32)
        im_tmp[:,:,0] = gt_img_np[:,:]
        im_tmp[:,:,1] = gt_img_np[:,:]
        im_tmp[:,:,2] = gt_img_np[:,:]
        gt_img_np =  im_tmp
        gt_img_np = gt_img_np.transpose((2, 0, 1))
        gt_img_np = torch.tensor(gt_img_np)
        '''
            
        return gt_img, rgb_img, gt_img_sparse
    def __len__(self):
        return len(self.samples)

class TestFolder(data.Dataset):

    def __init__(self, root,args, seed=None, train=True, transform=None, target_transform=None, mode = "DtoD"):
        np.random.seed(int(time.time()))                                                                    ## 랜덤하게 셔플하기 위한 장치
        random.seed(time.time())                                                                       ## 랜덤하게 셔플하기 위한 장치

        self.root = root
        self.transform = transform                                                              ## 매개변수로 받은 transform 클래스 대입
        self.train = train
        self.mode = mode
        self.args = args
        self.crawl_folders()                                                                    ## SequenceFolder 클래스의 crawl_folders 실행
        self.valid_index = 0
        self.num = 0

    def crawl_folders(self):                                                   
        sequence_set = []                                                                       ## 처음 sequence_set을 빈 list로 초기화
        imgs = []
        gt_depth = []
        gt_depth_color =[]
        with open(self.root + '/eigen_test_files_img.txt') as f:
            lines = f.readlines()
            for line in lines:
                line_ = line.split()
                imgs.append(line_[0])
        with open(self.root + '/eigen_test_files_color_gt.txt') as f:
            lines = f.readlines()
            for line in lines:
                line_ = line.split()
                gt_depth_color.append(line_[0])
        with open(self.root + '/eigen_test_files_gt.txt') as f:
            lines = f.readlines()
            for line in lines:
                line_ = line.split()
                gt_depth.append(line_[0])
        for i in range(len(imgs)):
            sample = {'gt': gt_depth[i], 'rgb': imgs[i], 'gt_color': gt_depth_color[i]}
            sequence_set.append(sample)
        self.samples = sequence_set                                                             ## self.samples에 셔플이 완료된 sequence_set 담기

    def __getitem__(self, index):
        self.valid_index = index

        while True:
            try:
                #print("sample[index]: ",self.samples[self.valid_index])
                sample = self.samples[self.valid_index]                                                            ## 주어지는 index에 해당하는 samples안의 이미지,gt pair data 추출
                gt_img = load_as_float(sample['gt_color'])                                                    ## sample에서 'gt이미지'에 해당하는 경로의 이미지 파일을 로드                                   
                rgb_img = load_as_float(sample['rgb'])
                gt_img_sparse = load_as_float(sample['gt'])
                #gt_img_sparse = depth_read(sample['gt'])
                #gt_img_sparse = cv2.resize(gt_img_sparse,(416,128),interpolation=cv2.INTER_AREA)
                
                if self.transform is not None:
                    imgs = self.transform([gt_img] + [rgb_img] + [gt_img_sparse])
                    gt_img = imgs[0]
                    rgb_img = imgs[1]
                    gt_img_sparse = imgs[2]
                break
            except:
                self.num = self.num + 1
                self.valid_index = self.valid_index - 1
                print('except된 파일: ',sample)
                print('except된 갯수: ',self.num)
                print('valid index: ',self.valid_index)
        '''
        gt_img_np = np.load(sample['gt_np']).astype(np.float32)
        im_tmp = np.zeros((128,416,3),dtype=np.float32)
        im_tmp[:,:,0] = gt_img_np[:,:]
        im_tmp[:,:,1] = gt_img_np[:,:]
        im_tmp[:,:,2] = gt_img_np[:,:]
        gt_img_np =  im_tmp
        gt_img_np = gt_img_np.transpose((2, 0, 1))
        gt_img_np = torch.tensor(gt_img_np)
        '''
            
        return gt_img, rgb_img, gt_img_sparse
    def __len__(self):
        return len(self.samples)

class TestFolder_file(data.Dataset):

    def __init__(self, root,args, seed=None, train=True, transform=None, target_transform=None, mode = "DtoD"):
        np.random.seed(int(time.time()))                                                                    ## 랜덤하게 셔플하기 위한 장치
        random.seed(time.time())                                                                       ## 랜덤하게 셔플하기 위한 장치

        self.root = root
        self.transform = transform                                                              ## 매개변수로 받은 transform 클래스 대입
        self.train = train
        self.mode = mode
        self.args = args
        self.resize = Resize()
        self.crawl_folders()                                                                    ## SequenceFolder 클래스의 crawl_folders 실행
        self.valid_index = 0
        self.num = 0

    def crawl_folders(self):                                                   
        sequence_set = []                                                                       ## 처음 sequence_set을 빈 list로 초기화
        imgs = []
        gt_depth = []
        gt_depth_color =[]
        
        
        with open(self.root + '/eigen_test_files_img.txt') as f:
            lines = f.readlines()
            for line in lines:
                line_ = line.split()
                imgs.append(line_[0])
        with open(self.root + '/eigen_test_files_color_gt.txt') as f:
            lines = f.readlines()
            for line in lines:
                line_ = line.split()
                gt_depth_color.append(line_[0])
        with open(self.root + '/eigen_test_files_gt.txt') as f:
            lines = f.readlines()
            for line in lines:
                line_ = line.split()
                gt_depth.append(line_[0])
        for i in range(len(imgs)):
            sample = {'gt': gt_depth[i], 'rgb': imgs[i], 'gt_color': gt_depth_color[i]}
            sequence_set.append(sample)
        '''
        large_imgs_path = Path('/mnt/MS/AEdepth/data_backup/test_input_1242')
        large_imgs = sorted(large_imgs_path.files('*.png'))
        for i in range(len(large_imgs)):
            sample = {'gt': large_imgs[i], 'rgb': large_imgs[i], 'gt_color': large_imgs[i]}
            sequence_set.append(sample)
        ########### Large size input ##############
        '''
        self.samples = sequence_set                                                             ## self.samples에 셔플이 완료된 sequence_set 담기
    def __getitem__(self, index):
        self.valid_index = index
        sample = self.samples[self.valid_index]                                                       ## 주어지는 index에 해당하는 samples안의 이미지,gt pair data 추출
        ''' to extract filename '''
        output_filename_list = sample['rgb'].split('/')
        output_filename = output_filename_list[5] + '_' + output_filename_list[6]
        output_filename = '/' + output_filename[:-4] + '.png'

        #print("sample[index]: ",self.samples[self.valid_index])
        gt_img = load_as_float(sample['gt_color'])                                                    ## sample에서 'gt이미지'에 해당하는 경로의 이미지 파일을 로드                                   
        rgb_img = load_as_float(sample['rgb'])
        gt_img_sparse = load_as_float(sample['gt'])

        #gt_img_sparse = depth_read(sample['gt'])
        #gt_img_sparse = cv2.resize(gt_img_sparse,(416,128),interpolation=cv2.INTER_AREA)
        '''
        gt_img = self.resize(gt_img,(368, 1232),'rgb')
        rgb_img = self.resize(rgb_img,(368, 1232),'rgb')
        if self.transform is not None:
            imgs = self.transform([gt_img] + [rgb_img] + [gt_img_sparse])
            gt_img = imgs[0]
            rgb_img = imgs[1]
            gt_img_sparse = imgs[2]
        '''
        
        while True:
            try:
                #print("sample[index]: ",self.samples[self.valid_index])
                gt_img = load_as_float(sample['gt_color'])                                                    ## sample에서 'gt이미지'에 해당하는 경로의 이미지 파일을 로드                                   
                rgb_img = load_as_float(sample['rgb'])
                gt_img_sparse = load_as_float(sample['gt'])

                #gt_img_sparse = depth_read(sample['gt'])
                #gt_img_sparse = cv2.resize(gt_img_sparse,(416,128),interpolation=cv2.INTER_AREA)
                
                if self.transform is not None:
                    imgs = self.transform([gt_img] + [rgb_img] + [gt_img_sparse])
                    gt_img = imgs[0]
                    rgb_img = imgs[1]
                    gt_img_sparse = imgs[2]
                break
            except:
                self.num = self.num + 1
                self.valid_index = self.valid_index - 1
                print('except된 파일: ',sample)
                print('except된 갯수: ',self.num)
                print('valid index: ',self.valid_index)
        
        '''
        gt_img_np = np.load(sample['gt_np']).astype(np.float32)
        im_tmp = np.zeros((128,416,3),dtype=np.float32)
        im_tmp[:,:,0] = gt_img_np[:,:]
        im_tmp[:,:,1] = gt_img_np[:,:]
        im_tmp[:,:,2] = gt_img_np[:,:]
        gt_img_np =  im_tmp
        gt_img_np = gt_img_np.transpose((2, 0, 1))
        gt_img_np = torch.tensor(gt_img_np)
        '''
            
        #return gt_img, rgb_img, gt_img_sparse
        return gt_img, rgb_img, output_filename
    def __len__(self):
        return len(self.samples)

class Make3DFolder(data.Dataset):

    def __init__(self, root,args, seed=None, train=True, transform=None, target_transform=None, mode = "DtoD"):
        np.random.seed(int(time.time()))                                                                    ## 랜덤하게 셔플하기 위한 장치
        random.seed(time.time())                                                                       ## 랜덤하게 셔플하기 위한 장치
        self.root = Path(root)                                                              ## Dataset의 경로 path설정
        self.mat_depth_folder = self.root/'Train400Depth' if train else self.root/'Test134Depth'               ## Dataset의 폴더 리스트가 담긴 txt파일 경로
        self.depth_folder = self.root/'Train400Depth_img_bmp_size2' if train else self.root/'Test134Depth_img_bmp_size2'
        self.img_folder = self.root/'Train400' if train else self.root/'Test134'
        self.transform = transform                                                              ## 매개변수로 받은 transform 클래스 대입
        self.train = train
        self.mode = mode
        self.args = args
        self.crawl_folders()                                                                    ## SequenceFolder 클래스의 crawl_folders 실행

    def crawl_folders(self):                                                   
        sequence_set = []                                                                 ## 처음 sequence_set을 빈 list로 초기화
        gt_depth = sorted(self.depth_folder.files('*.bmp'))
        rgbs = sorted(self.img_folder.files('*.jpg'))                                                 ## input 이미지들을 정렬
        #gt_depth_np = sorted(mat_depth_folder.files('*.mat'))
        #gt_depth_np = sorted(scene.files('*.npy'))
        gt_depth_np = sorted(self.depth_folder.files('*.bmp'))
        for i in range(len(rgbs)):
            sample = {'gt': gt_depth[i], 'rgb': rgbs[i], 'gt_np': gt_depth_np[i]}
            sequence_set.append(sample)
        if (self.args.img_test is False) or (self.train is True):
            random.shuffle(sequence_set)                                                            ## 모두 추가된 sequence_set를 셔플
        self.samples = sequence_set                                                             ## self.samples에 셔플이 완료된 sequence_set 담기

    def __getitem__(self, index):
        sample = self.samples[index]
        try:                                                           ## 주어지는 index에 해당하는 samples안의 이미지,gt pair data 추출
            gt_img = load_as_float(sample['gt'])
        except:                                                  ## sample에서 'gt이미지'에 해당하는 경로의 이미지 파일을 로드                                   
            print('로드에러: ',sample['gt'],'의 이미지 로드 에러')
        try:
            rgb_img = load_as_float(sample['rgb'])
        except:
            print('로드에러: ',sample['rgb'],'의 이미지 로드 에러')
        gt_img_sparse = load_as_float(sample['gt_np'])
        rgb_img = cv2.resize(rgb_img,(176,232),interpolation=cv2.INTER_AREA)
        if self.transform is not None:
            imgs = self.transform([gt_img] + [rgb_img] + [gt_img_sparse])
            gt_img = imgs[0]
            rgb_img = imgs[1]
            gt_img_sparse = imgs[2]
        '''
        gt_img_np = np.load(sample['gt_np']).astype(np.float32)
        im_tmp = np.zeros((128,416,3),dtype=np.float32)
        im_tmp[:,:,0] = gt_img_np[:,:]
        im_tmp[:,:,1] = gt_img_np[:,:]
        im_tmp[:,:,2] = gt_img_np[:,:]
        gt_img_np =  im_tmp
        gt_img_np = gt_img_np.transpose((2, 0, 1))
        gt_img_np = torch.tensor(gt_img_np)
        '''
            
        return gt_img, rgb_img, gt_img_sparse
    def __len__(self):
        return len(self.samples)

class NYUdataset(data.Dataset):
    def __init__(self, root,args, seed=None, train=True, transform=None, transform_2=None, mode = "DtoD"):
        np.random.seed(int(time.time()))                                                                    ## 랜덤하게 셔플하기 위한 장치
        random.seed(time.time())                                                                       ## 랜덤하게 셔플하기 위한 장치
        self.root = Path(root)                                                              ## Dataset의 경로 path설정
        self.depth_folder = self.root/'train/train_depths' if train else self.root/'test/test_depths'               ## Dataset의 폴더 리스트가 담긴 txt파일 경로
        self.img_folder = self.root/'train/train_colors' if train else self.root/'test/test_colors'
        self.transform = transform 
        self.transform_2 = transform_2                                                             ## 매개변수로 받은 transform 클래스 대입
        self.train = train
        self.mode = mode
        self.args = args
        self.resize = imresize
        self.crawl_folders()                                                                    ## SequenceFolder 클래스의 crawl_folders 실행

    def crawl_folders(self):                                                   
        sequence_set = []                                                                 ## 처음 sequence_set을 빈 list로 초기화
        gt_depth = sorted(self.depth_folder.files('*.png'))
        rgbs = sorted(self.img_folder.files('*.png'))
        for i in range(len(rgbs)):
            sample = {'gt': gt_depth[i], 'rgb': rgbs[i]}
            sequence_set.append(sample)
        if (self.args.img_test is False) or (self.train is True):
            random.shuffle(sequence_set)                                                            ## 모두 추가된 sequence_set를 셔플
        self.samples = sequence_set                                                             ## self.samples에 셔플이 완료된 sequence_set 담기

    def __getitem__(self, index):
        sample = self.samples[index]
        #gt_img = Image.open(sample['gt'])
        #rgb_img = Image.open(sample['rgb'])
        gt_img = load_as_float(sample['gt'])
        rgb_img = load_as_float(sample['rgb'])
        
        if self.train is True:
            ## train_transform applied
            img_s = np.random.uniform(1,1.2)
            scale = np.random.uniform(1.0, 1.5)
            gt_img = np.array(gt_img, dtype=np.float32)
            gt_img = self.resize(gt_img,(int(img_s*251.0), int(img_s*340.0)),'depth')
            rgb_img = np.array(rgb_img, dtype=np.float32) 
            ## if RtoD train, color image get rotated and randomcrop also, but in DtoD mode only depth image transformed
            if self.args.mode == 'DtoD':
                rgb_img = self.resize(rgb_img,(251,340),'rgb')
            else :
                rgb_img = self.resize(rgb_img,(int(img_s*251.0), int(img_s*340.0)),'rgb')
            gt_img = gt_img / scale
            if(gt_img.ndim==2):
                gt_img_tmp = np.zeros((int(img_s*251.0), int(img_s*340.0),1),dtype=np.float32)
                gt_img_tmp[:,:,0] = gt_img[:,:]
                gt_img = gt_img_tmp
            if self.args.mode == 'DtoD':
                imgs = self.transform([gt_img])
                imgs[0] = self.resize(imgs[0],scale,'depth')
                imgs = [rgb_img] + imgs ## [rgb, gt_depth]
            else:
                imgs = self.transform([rgb_img] + [gt_img])
                imgs[0] = self.resize(imgs[0],scale,'rgb')
                imgs[1] = self.resize(imgs[1],scale,'depth')
            imgs = self.transform_2(imgs)
        else:
            ## valid_trasform applied
            gt_img = np.array(gt_img, dtype=np.float32)
            #gt_img = self.resize(gt_img,(251,340),'depth')
            ##gt_img = self.resize(gt_img,(251,340))
            rgb_img = np.array(rgb_img, dtype=np.float32)
            #rgb_img = self.resize(rgb_img,(251,340),'rgb')
            ##rgb_img = self.resize(rgb_img,(251,340))
            if(gt_img.ndim==2):
                ##gt_img_tmp = np.zeros((251, 340 ,1),dtype=np.float32)
                gt_img_tmp = np.zeros((320, 420 ,1),dtype=np.float32)
                gt_img_tmp[:,:,0] = gt_img[:,:]
                gt_img = gt_img_tmp
            imgs = self.transform([rgb_img] + [gt_img])
        rgb_img = imgs[0]
        gt_img = imgs[1]
            
        return gt_img, rgb_img, gt_img
    def __len__(self):
        return len(self.samples)