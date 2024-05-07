import os
import torch
import random
import os.path
import numpy as np
import torch.utils.data as data
from skimage.transform import resize
from torchvision.datasets.folder import is_image_file
from utils import color
import cv2

def default_loader(path, exr=False, npy=False, max_val=1., min_val=0.):
    try:
        if exr:
            img = cv2.imread(path, 2)
            img[img > max_val] = max_val
            img[img < min_val] = min_val
            img = (img - min_val) / ((max_val - min_val))
            return img
        elif npy:
            img = np.load(path)
            return img
        else:
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print(path)

def make_dataset(root, file_name, type, num):
    images = []
    if type:
        pwd_file = os.path.join(root, file_name + '_' + type)
    else:
        pwd_file = os.path.join(root, file_name)

    if os.path.isfile(pwd_file):
        with open(pwd_file, 'r') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip()
                images.append(os.path.join(root, line))
    else:
        assert os.path.isdir(pwd_file), f'{pwd_file} must be a folder or file.'
        names = os.listdir(pwd_file)
        images = [os.path.join(pwd_file, x) for x in names if is_image_file(x)]
    
    sorted(images)
    if num:
        images = images[:num]
    return images

class LowLevelVisionFolder(data.Dataset):
    def __init__(self, params, 
                loader=default_loader, image_names=None, is_train=True, type='') -> None:
        super().__init__()
        self.image_names    = image_names
        self.root           = params.data_root
        self.loader         = loader
        self.height         = params.image_height
        self.width          = params.image_width
        self.rotation_range = 15.0 
        self.is_train       = is_train
        self.num = params.num_limit

        self.trunc_max = 0.8
        self.trunc_min = 0.3

        self.type = type

        img_dict = {}
        for i in self.image_names:
            img_dict[i] = make_dataset(self.root, i, self.type, self.num)

        assert all([len(x) == len(img_dict[image_names[0]]) for x in img_dict.values()])
        
        self.image_dict = img_dict
    
    def __len__(self) -> int:
        return len(self.image_dict[self.image_names[0]])
    
    def data_augments(self, img, mode, random_pos):

        img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3]]
        img = resize(img, (self.height, self.width), order=mode, preserve_range=True)

        return img
    
    def load_images(self, index, use_da=True):
        images = {}

        for k in self.image_names:
            images[k] = self.loader(self.image_dict[k][index], exr=(k=='depth-exr'), npy=(k in ['segmentation', 'lighting']), max_val=self.trunc_max, min_val=self.trunc_min)

            if k == 'normal':
                images[k] = images[k].astype(np.float32)
                images[k] /= 255.
                norm = np.linalg.norm(images[k], axis=2) + 1e-6 
                images[k] = images[k] / np.expand_dims(norm, axis=-1)

                continue

            if k in ['edge', 'mask', 'peripheral', 'eyebrow', 'skin']:
                images[k] = images[k].astype(np.float32)
                images[k] /= 255.
                continue

            if k == 'lighting':
                continue

            images[k] = images[k].astype(np.float32)
            images[k] /= 255.
            images[k] = np.vectorize(color.srgb2lin)(images[k])
        images['chromaticity'] = color.rgb_to_chromaticity(images['composite'])
        images['mask-inner'] = np.ones([self.height, self.width, 1])
        file_name = os.path.basename(self.image_dict[self.image_names[0]][index])
        ori_h, ori_w = images[self.image_names[0]].shape[:2]

        if use_da: 
            random_start_x = random.randint(0, 9)
            random_start_y = random.randint(0, 9)

            random_pos = [random_start_y, random_start_y + ori_h - 10, random_start_x,
                          random_start_x + ori_w - 10]
            
            for k in images.keys():
                if k == 'lighting':
                    continue
                elif k == 'segmentation' or k == 'normal':
                    inter_type = 0
                else:
                    inter_type = 1
                images[k] = self.data_augments(images[k], mode=inter_type, random_pos=random_pos) 
        else:
            for k in images.keys():
                if k == 'lighting':
                    continue
                elif k == 'segmentation' or k == 'normal': 
                    inter_type = 0
                else:
                    inter_type = 1
                images[k] = resize(images[k], (self.height, self.width), order=inter_type, preserve_range=True)

        images['filename'] = file_name

        return images
    
    def __getitem__(self, index: int):
        images = self.load_images(index, use_da=self.is_train)

        for k in images.keys():
            if type(images[k]) is str: # filename
                continue
            if k == 'segmentation':
                images[k] = torch.from_numpy(np.transpose(images[k], (2, 0, 1))).long().contiguous()
                continue
            elif k == 'lighting':
                images[k] = torch.from_numpy(images[k].reshape(-1)).contiguous()
                continue

            images[k][images[k] < 1e-4] = 1e-4
            images[k] = torch.from_numpy(np.transpose(images[k], (2, 0, 1))).contiguous().float()
        return images