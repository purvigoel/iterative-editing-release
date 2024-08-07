from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, RandomCrop, ToPILImage, ToTensor, Resize
import numpy as np
import pickle
import cv2
from os.path import join
import os
import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
print('Setting np random seed: 0')
np.random.seed(0)
import glob

def unnormalize_img(img):
    """For a single image tensor."""
    nimg = img.clone()
    nimg[0].mul_(constants.IMG_NORM_STD[0]).add_(
        constants.IMG_NORM_MEAN[0])
    nimg[1].mul_(constants.IMG_NORM_STD[1]).add_(
        constants.IMG_NORM_MEAN[1])
    nimg[2].mul_(constants.IMG_NORM_STD[2]).add_(
        constants.IMG_NORM_MEAN[2])
    return nimg

def tensor2image(tensor):
    """For a single image tensor."""
    assert len(tensor.shape) == 3
    assert tensor.shape[0] == 3
    return tensor.permute(1,2,0).numpy()

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True, parse_random_bg=False):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(
            mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.normalize_img = Normalize(
            mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']
        # self.transl = self.data['trans']
        self.parse_random_bg = parse_random_bg

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate(
            [keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array(
                [0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train and self.use_augmentation:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(
                1-self.options.noise_factor, 1+self.options.noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                      max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                     max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                       [constants.IMG_RES, constants.IMG_RES], rot=rot,
                       edge_padding=False)  # make this an argument?
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(
            255.0, np.maximum(0.0, rgb_img[:, :, 0]*pn[0]))
        rgb_img[:, :, 1] = np.minimum(
            255.0, np.maximum(0.0, rgb_img[:, :, 1]*pn[1]))
        rgb_img[:, :, 2] = np.minimum(
            255.0, np.maximum(0.0, rgb_img[:, :, 2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2]+1, center, scale,
                                   [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2.*kp[:, :-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = self.imgname[index]
        #    imgname = join('S9', 'Videos', imgname[10:].replace('_', ' '))
        # elif imgname.startswith('images/S11_'):
        #    imgname = join('S11', 'Videos', imgname[11:].replace('_', ' '))

        # Load EFT pseudo graoundtruth
        if 'eft' in dir(self.options) and self.options.eft and not self.options.eft_no_psuedogt:
            # pretty adhoc implementation ...
            assert self.options.ft_dataset in ['3dpw_test_eft', '3dpw_train_eft']
            PW3D_EFT_ROOT = '/home/users/wangkua1/projects/eft/eft_out'
            eft_pickle_dir = {
                '3dpw_train_eft': os.path.join(PW3D_EFT_ROOT, 'eftout_01-28-80938_3dpw_train_3dpw_train'),
                '3dpw_test_eft':  os.path.join(PW3D_EFT_ROOT, 'eftout_01-28-81012_3dpw_test_3dpw_train')
            }[self.options.ft_dataset]
            # Example imgname: imageFiles/downtown_enterShop_00/image_00000.jpg
            _, dirname, filename = imgname.split('.')[0].split('/')
            suffix = str(int(filename.split('_')[-1]))  # e.g. 00000
            # pklname = f'{dirname}_{filename}_{suffix}.pkl'
            # path = os.path.join(eft_pickle_dir, pklname)
            pklprefix = f'{dirname}_{filename}'
            # paths = glob.glob(os.path.join(eft_pickle_dir, f'{pklprefix}*'))
            # import ipdb; ipdb.set_trace()
            path = os.path.join(eft_pickle_dir, f'{dirname}_{filename}_{index}.pkl')
            # if len(paths) > 0:
                # path = paths[0]
            if os.path.exists(path):
                stuff = pickle.load(open(path, 'rb'))
                item['eft_rotmat'] = torch.tensor(stuff['pred_pose_rotmat'])
                item['eft_shape'] = torch.tensor(stuff['pred_shape'])
                item['eft_camera'] = torch.tensor(stuff['pred_camera'])
                item['eft_exists'] = 1
            else:
                item['eft_exists'] = 0

        imgname = join(self.img_dir, imgname)
        try:
            img = cv2.imread(imgname)[:, :, ::-1].copy()
            if self.parse_random_bg:
                random_bg = RandomCrop(
                    scale*200, pad_if_needed=True, padding_mode='edge')(ToPILImage()(img))
                random_bg = ToTensor()(Resize((constants.IMG_RES, constants.IMG_RES))(random_bg))
#                 assert random_bg.shape[0] == 3, random_bg.shape
#                 start_x, end_x = 0, 0
#                 start_y, end_y = center[1] - 100, center[1] + 100
#                 x = np.random.randint(start_x, max(start_x, end_x-scale*200))
#                 y = np.random.randint(start_y, max(start_y, end_y-scale*200))

            img = img.astype(np.float32)
        except:
            print(imgname)
            os.exit(0)

        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        original_image = img.copy()
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(
            self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        if self.parse_random_bg:
            # appy the same color jittering as the foreground.
            random_bg[0, :, :] = np.minimum(1.0, np.maximum(0.0, random_bg[0,:,:]*pn[0]))
            random_bg[1, :, :] = np.minimum(1.0, np.maximum(0.0, random_bg[1,:,:]*pn[1]))
            random_bg[2, :, :] = np.minimum(1.0, np.maximum(0.0, random_bg[2,:,:]*pn[2]))
            item['random_bg'] = random_bg

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            original_pose_3d = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(
                self.j3d_processing(S, rot, flip)).float()
            item['original_pose_3d'] = original_pose_3d
        else:
            item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        original_keypoints = self.keypoints[index].copy()
        keypoints = self.keypoints[index].copy()
        item['original_image'] = original_image
        item['original_keypoints'] = original_keypoints
        item['keypoints'] = torch.from_numpy(self.j2d_processing(
            keypoints, center, sc*scale, rot, flip)).float()

        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)
