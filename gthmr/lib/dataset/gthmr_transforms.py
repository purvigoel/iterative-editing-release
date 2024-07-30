"""
Transformations needed by the Global HMR problem. 

Naming conventions: 
    `rf` -- "reference frame"
    `zif` -- "zero initial frame" which means having (0, 0, 0) aa root rotation and (0, 0, 0) xyz root translation

A sequence lives in some <REFERENCE-FRAME> (RF) ... 

RF: "World", or "Camera"
    "World" RF has a ground plane aligned with the axes, and "Camera" RF does not. 
    

"""
import os
import os.path as osp
import sys
import cv2
import glob
import h5py
import pickle as pkl
import ipdb
import numpy as np
import argparse
from tqdm import tqdm
import cdflib

# VIBE related
from lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR, H36M_DIR

# Viz
import torch
from utils.geometry import perspective_projection, perspective_projection_with_K

mosh_dir = osp.join(H36M_DIR, 'mosh/neutrMosh/neutrSMPL_H3.6/')
test_mosh = osp.join(mosh_dir, "S1", "Walking_cam0_aligned.pkl")





def get_action_name_from_action_id(s):
    return s.split(' ')[0]


def action_id_without_chair(s):
    actions_with_chair = set([
        get_action_name_from_action_id(s) for s in ACTIONS
    ]).difference(
        set([get_action_name_from_action_id(s)
             for s in ACTIONS_WITHOUT_CHAIR]))

    return get_action_name_from_action_id(s) not in actions_with_chair


def action_id_without_chair0(s):
    raise
    # this is not good... use the one above...
    return get_action_name_from_action_id(s) in set(
        [get_action_name_from_action_id(s) for s in ACTIONS_WITHOUT_CHAIR])


def h36m_train_extract_abs_root_loc_given_name(dataset_path, id, frame_rate=5):
    """
    example id: "S1_WalkingDog_1.54138969" "S{user_id}_{action}">
    """
    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    prefix = id
    idx = prefix.find('_')
    user_i = int(prefix[1:idx])
    action = prefix[idx + 1:].replace('_', ' ')
    print('User:', user_i)
    user_name = 'S%d' % user_i
    # path with GT 3D pose
    pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                             'D3_Positions')
    seq_name = f"{action}.cdf"
    seq_i = os.path.join(pose_path, seq_name)
    print('\tSeq:', seq_i)
    sys.stdout.flush()
    # irrelevant sequences
    if action == '_ALL':
        return

    # 3D pose file
    poses_3d = cdflib.CDF(seq_i)['Pose'][0]
    root_loc_vec = []
    # go over each frame of the sequence
    for frame_i in range(poses_3d.shape[0]):
        protocol = 1
        if frame_i % frame_rate == 0 and (protocol == 1
                                          or camera == '60457274'):
            # read GT 3D pose
            Sall = np.reshape(poses_3d[frame_i, :], [-1, 3]) / 1000.
            S17 = Sall[h36m_idx]
            root_loc_vec.append(S17[0])
    return np.array(root_loc_vec)


if __name__ == '__main__':
    """
    """
    from nemo.utils.render_utils import add_keypoints_to_image
    
    out_dir = '_h36m'
    os.makedirs(out_dir, exist_ok=True)

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # Output
    dataset = {
        'img_name': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'features': [],
    }

    user_list = [1]

    # go over each user
    for user_i in user_list:
        print('User:', user_i)
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat',
                                 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        # path with GT 3D pose2
        pose2_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                  'D3_Positions')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                   'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in seq_list:
            print('\tSeq:', seq_i)
            sys.stdout.flush()
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action_w_space, camera, _ = seq_name.split('.')
            action = action_w_space.replace(' ', '_')

            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = cdflib.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # Mosh
            cam_id = CAMERAS.index(camera)
            mosh_path = osp.join(mosh_dir, user_name,
                                 f"{action_w_space}_cam{cam_id}_aligned.pkl")
            mosh = pkl.load(open(mosh_path, 'rb'), encoding="latin1")

            # go over each frame of the sequence
            for frame_i in tqdm(range(poses_3d.shape[0])):
                protocol = 1
                if frame_i % 5 == 0 and (protocol == 1
                                         or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera,
                                                     frame_i + 1)

                    img_path = osp.join(dataset_path, 'images', imgname)

                    # read GT bounding box
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i, 0]][()].T
                    ys, xs = np.where(mask == 1)
                    bbox = np.array([
                        np.min(xs),
                        np.min(ys),
                        np.max(xs) + 1,
                        np.max(ys) + 1
                    ])
                    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                    scale = 0.9 * max(bbox[2] - bbox[0],
                                      bbox[3] - bbox[1]) / 200.

                    # read GT 2D pose
                    partall = np.reshape(poses_2d[frame_i, :], [-1, 2])
                    part17 = partall[h36m_idx]
                    part = np.zeros([24, 3])
                    part[global_idx, :2] = part17
                    part[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i, :], [-1, 3]) / 1000.
                    S17 = Sall[h36m_idx]
                    # S17 -= S17[0] # root-centered
                    S24 = np.zeros([24, 4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1

                    # Viz
                    im = cv2.imread(img_path)
                    camera_rotation = torch.eye(3).unsqueeze(0).expand(
                        1, -1, -1)
                    camera_translation = torch.zeros(1, 3)
                    K = torch.load(
                        '/home/users/wangkua1/projects/bio-pose/camera_intrinsics.pt'
                    )
                    projected_keypoints_2d = perspective_projection_with_K(
                        torch.tensor(S17)[None].float(),
                        rotation=camera_rotation,
                        translation=camera_translation,
                        K=K).detach().numpy()[0]
                    
                    ipdb.set_trace()

                    # store data
                    dataset['img_name'].append(os.path.join('images', imgname))
                    dataset['joints3D'].append(j3d)
                    dataset['joints3D'].append(j2d)
                    dataset['shape'].append(shape)
                    dataset['pose'].append(pose)
                    dataset['bbox'].append(bbox)
                    dataset['features'].append(features)

    # store the data struct
    out_file = os.path.join(VIBE_DB_DIR, 'h36m_dev_walking.npz')
