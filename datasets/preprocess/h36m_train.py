import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from tqdm import tqdm
#from spacepy import pycdf
import cdflib
# from .read_openpose import read_openpose

ACTIONS = [
    "Directions 1", "Directions", "Discussion 1", "Discussion", "Eating 2",
    "Eating", "Greeting 1", "Greeting", "Phoning 1", "Phoning", "Posing 1",
    "Posing", "Purchases 1", "Purchases", "Sitting 1", "Sitting 2",
    "SittingDown 2", "SittingDown", "Smoking 1", "Smoking", "TakingPhoto 1",
    "TakingPhoto", "Waiting 1", "Waiting", "Walking 1", "Walking",
    "WalkingDog 1", "WalkingDog", "WalkTogether 1", "WalkTogether"
]

ACTIONS_WITHOUT_CHAIR = [
    "Directions 1", "Directions", "Discussion 1", "Discussion", "Greeting 1",
    "Greeting", "Posing 1", "Posing", "Purchases 1", "Purchases",
    "SittingDown 2", "SittingDown", "TakingPhoto 1", "TakingPhoto",
    "Waiting 1", "Waiting", "Walking 1", "Walking", "WalkingDog 1",
    "WalkingDog", "WalkTogether 1", "WalkTogether"
]

CAMERAS = ["54138969", "55011271", "58860488", "60457274"]


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


def h36m_train_extract_root_loc_given_name(dataset_path, id, frame_rate=5):
    """
    example id: "S1_WalkingDog_1.54138969" "S{user_id}_{action}.{camera}">
    """
    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    prefix, camera = id.split('.')
    idx = prefix.find('_')
    user_i = int(prefix[1:idx])
    action = prefix[idx + 1:].replace('_', ' ')
    print('User:', user_i)
    user_name = 'S%d' % user_i
    # path with GT 3D pose
    pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                             'D3_Positions_mono')
    seq_name = f"{action}.{camera}.cdf"
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


def h36m_train_extract(dataset_path,
                       sn,
                       action,
                       camera,
                       frame_rate=5,
                       extract_img=False):
    """
    Main differences compared to the original code:
      - include absolute 3D poses (i.e., agnostic to camera view)
      - include root information on all 3D info (i.e., not root-centered)

    """
    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, Ss_abs_ = [], [], [], [], [], []

    print('User:', sn)
    user_name = sn
    # path with GT bounding boxes
    bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat',
                             'ground_truth_bb')
    # path with GT 3D pose
    pose_abs_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions')
    # path with GT 3D pose
    pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                             'D3_Positions_mono')
    # path with GT 2D pose
    pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                               'D2_Positions')
    # path with videos
    vid_path = os.path.join(dataset_path, user_name, 'Videos')

    # go over all the sequences of each user
    seq_name = f'{action}.{camera}.cdf'

    print('\tSeq:', seq_name)
    sys.stdout.flush()
    # sequence info
    action_1, camera, _ = seq_name.split('.')
    action = action_1.replace(' ', '_')

    # 3D pose file
    seq_i = os.path.join(pose_path, seq_name)
    poses_3d = cdflib.CDF(seq_i)['Pose'][0]

    # Abs 3D pose file
    pose_abs_file = os.path.join(pose_abs_path, f"{action_1}.cdf")
    poses_abs = cdflib.CDF(pose_abs_file)['Pose'][0]

    # 2D pose file
    pose2d_file = os.path.join(pose2d_path, seq_name)
    poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

    # bbox file
    bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
    if os.path.exists(bbox_file):
        bbox_h5py = h5py.File(bbox_file)

    # video file
    if extract_img:
        vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
        imgs_path = os.path.join(dataset_path, 'images')
        vidcap = cv2.VideoCapture(vid_file)

    # go over each frame of the sequence
    for frame_i in range(poses_3d.shape[0]):
        # read video frame
        if extract_img:
            success, image = vidcap.read()
            if not success:
                break
        # check if you can keep this frame
        protocol = 1
        if frame_i % frame_rate == 0 and (protocol == 1
                                          or camera == '60457274'):
            # image name
            imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera,
                                             frame_i + 1)

            # save image
            if extract_img:
                img_out = os.path.join(imgs_path, imgname)
                cv2.imwrite(img_out, image)

            if os.path.exists(bbox_file):
                # read GT bounding box
                mask = bbox_h5py[bbox_h5py['Masks'][frame_i, 0]][()].T
                ys, xs = np.where(mask == 1)
                bbox = np.array(
                    [np.min(xs),
                     np.min(ys),
                     np.max(xs) + 1,
                     np.max(ys) + 1])
                center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                scale = 0.9 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

                centers_.append(center)
                scales_.append(scale)

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

            # read GT Absolute 3D pose
            Sall = np.reshape(poses_abs[frame_i, :], [-1, 3]) / 1000.
            S17 = Sall[h36m_idx]
            # S17 -= S17[0] # root-centered
            S24_abs = np.zeros([24, 4])
            S24_abs[global_idx, :3] = S17
            S24_abs[global_idx, 3] = 1

            # # read openpose detections
            # json_file = os.path.join(openpose_path,
            #     imgname.replace('.jpg', '_keypoints.json'))
            # openpose = read_openpose(json_file, part, 'h36m')

            # store data
            imgnames_.append(os.path.join('images', imgname))
            parts_.append(part)
            Ss_.append(S24)
            Ss_abs_.append(S24_abs)
    return {
        'imgnames': imgnames_,
        '2d_keypoints': np.array(parts_),
        '3d_keypoints': np.array(Ss_),
        '3d_abs_keypoints': np.array(Ss_abs_),
    }
    # # store the data struct
    # if not os.path.isdir(out_path):
    #     os.makedirs(out_path)
    # out_file = os.path.join(out_path, 'h36m_train.npz')
    # np.savez(out_file, imgname=imgnames_,
    #                    center=centers_,
    #                    scale=scales_,
    #                    part=parts_,
    #                    S=Ss_,
    #                    S_abs=Ss_abs_)


# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract_original_example(dataset_path,
                                        openpose_path,
                                        out_path,
                                        extract_img=False):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, openposes_  = [], [], [], [], [], []

    # users in validation set
    user_list = [1, 5, 6, 7, 8]

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
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
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

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path,
                                        seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break
                # check if you can keep this frame
                protocol = 1
                if frame_i % 5 == 0 and (protocol == 1
                                         or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera,
                                                     frame_i + 1)

                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)
                        #continue
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

                    # read GT 3D pose
                    partall = np.reshape(poses_2d[frame_i, :], [-1, 2])
                    part17 = partall[h36m_idx]
                    part = np.zeros([24, 3])
                    part[global_idx, :2] = part17
                    part[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i, :], [-1, 3]) / 1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0]  # root-centered
                    S24 = np.zeros([24, 4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1

                    # read openpose detections
                    json_file = os.path.join(
                        openpose_path,
                        imgname.replace('.jpg', '_keypoints.json'))
                    openpose = read_openpose(json_file, part, 'h36m')

                    # store data
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(part)
                    Ss_.append(S24)
                    openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'h36m_train.npz')
    np.savez(out_file,
             imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             S=Ss_,
             openpose=openposes_)


if __name__ == '__main__':
    dataset_path = "/oak/stanford/groups/syyeung/hmr_datasets/h36m"
    out_path = '.'
    extract_img = True

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, openposes_  = [], [], [], [], [], []

    # users in validation set
    # user_list = [1, 5, 6, 7, 8]
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
            if 'walking' not in seq_i.lower() or 'dog' in seq_i.lower():
                continue

            print('\tSeq:', seq_i)
            sys.stdout.flush()
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')

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

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path,
                                        seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                os.makedirs(imgs_path, exist_ok=True)
                vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in tqdm(range(poses_3d.shape[0])):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break
                # check if you can keep this frame
                protocol = 1
                if frame_i % 5 == 0 and (protocol == 1
                                         or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera,
                                                     frame_i + 1)

                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)
                        #continue
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

                    # import ipdb; ipdb.set_trace()
                    # # read openpose detections
                    # json_file = os.path.join(openpose_path,
                    #     imgname.replace('.jpg', '_keypoints.json'))
                    # openpose = read_openpose(json_file, part, 'h36m')

                    # store data
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(part)
                    Ss_.append(S24)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'h36m_dev_walking.npz')
    np.savez(out_file,
             imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             S=Ss_,
             openpose=openposes_)

# if __name__ == '__main__':
#     dataset_path = "/scratch/groups/syyeung/hmr_datasets/h36m_train/"
#     extract_img = False

#     # convert joints to global order
#     h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
#     global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

#     # structs we use
#     imgnames_, scales_, centers_, parts_, Ss_, openposes_  = [], [], [], [], [], []

#     # users in validation set
#     # user_list = [1, 5, 6, 7, 8]
#     user_list = [1]

#     # go over each user
#     for user_i in user_list:
#         print('User:', user_i)
#         user_name = 'S%d' % user_i
#         # path with GT bounding boxes
#         bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
#         # path with GT 3D pose
#         pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
#         # path with GT 3D pose2
#         pose2_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions')
#         # path with GT 2D pose
#         pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
#         # path with videos
#         vid_path = os.path.join(dataset_path, user_name, 'Videos')

#         # go over all the sequences of each user
#         seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
#         seq_list.sort()
#         for seq_i in seq_list:
#             print('\tSeq:', seq_i)
#             sys.stdout.flush()
#             # sequence info
#             seq_name = seq_i.split('/')[-1]
#             action, camera, _ = seq_name.split('.')
#             action = action.replace(' ', '_')

#             # irrelevant sequences
#             if action == '_ALL':
#                 continue

#             # 3D pose file
#             poses_3d = cdflib.CDF(seq_i)['Pose'][0]

#             # 2D pose file
#             pose3d2_file = os.path.join(pose2_path, action.replace('_', ' '))
#             poses_3d2 = cdflib.cdfread.CDF(pose3d2_file)['Pose'][0]

#             # 2D pose file
#             pose2d_file = os.path.join(pose2d_path, seq_name)
#             poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

#             # bbox file
#             bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
#             bbox_h5py = h5py.File(bbox_file)

#             # video file
#             if extract_img:
#                 vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
#                 imgs_path = os.path.join(dataset_path, 'images')
#                 vidcap = cv2.VideoCapture(vid_file)

#             # go over each frame of the sequence
#             for frame_i in range(poses_3d.shape[0]):
#                 # read video frame
#                 if extract_img:
#                     success, image = vidcap.read()
#                     if not success:
#                         break
#                 # check if you can keep this frame
#                 protocol = 1
#                 if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
#                     # image name
#                     imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)

#                     # save image
#                     if extract_img:
#                         img_out = os.path.join(imgs_path, imgname)
#                         cv2.imwrite(img_out, image)
#                         #continue
#                     # read GT bounding box
#                     mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]][()].T
#                     ys, xs = np.where(mask==1)
#                     bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
#                     center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
#                     scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

#                     # read GT 3D pose
#                     partall = np.reshape(poses_2d[frame_i,:], [-1,2])
#                     part17 = partall[h36m_idx]
#                     part = np.zeros([24,3])
#                     part[global_idx, :2] = part17
#                     part[global_idx, 2] = 1

#                     # read GT 3D pose
#                     Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
#                     S17 = Sall[h36m_idx]
#                     S17 -= S17[0] # root-centered
#                     S24 = np.zeros([24,4])
#                     S24[global_idx, :3] = S17
#                     S24[global_idx, 3] = 1

#                     import ipdb; ipdb.set_trace()
#                     # # read openpose detections
#                     # json_file = os.path.join(openpose_path,
#                     #     imgname.replace('.jpg', '_keypoints.json'))
#                     # openpose = read_openpose(json_file, part, 'h36m')

#                     # store data
#                     imgnames_.append(os.path.join('images', imgname))
#                     centers_.append(center)
#                     scales_.append(scale)
#                     parts_.append(part)
#                     Ss_.append(S24)
#                     # openposes_.append(openpose)
