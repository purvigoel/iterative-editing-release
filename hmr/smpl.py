import torch
import numpy as np
import smplx
import ipdb
from smplx import SMPL as _SMPL
from smplx import SMPLX as _SMPLX
try:
    from smplx.body_models import ModelOutput
except:
    from smplx.body_models import SMPLOutput as ModelOutput
from smplx.lbs import vertices2joints

import hmr.hmr_config as config
import hmr.hmr_constants as constants


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            'J_regressor_extra',
            torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        # ipdb.set_trace()
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra,
                                       smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


class SMPLX(_SMPLX):
    """ Extension of the official SMPLX implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPLX, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            'J_regressor_extra',
            torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        # ipdb.set_trace()
        kwargs['get_skin'] = True
        smpl_output = super(SMPLX, self).forward(*args, **kwargs)
        # extra_joints = vertices2joints(self.J_regressor_extra,
        #                                smpl_output.vertices)
        # joints = torch.cat([smpl_output.joints[:, :45], extra_joints], dim=1)
        # joints = joints[:, self.joint_map, :]
        joints = smpl_output.joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


if __name__ == '__main__':
    """
    "__quick_viz_24_90_180_0" looks great!!!!!! (C=[0, 0, 20]), focal_length=10000
    looks great!!!!!! (R=[90, 160, 0], C=[0, 1.5, 5]), focal_length=5000

    """
    from hmr import hmr_config
    from hmr.renderer import Renderer
    from hmr.geometry import apply_extrinsics
    import cv2
    from itertools import product
    import torch
    import joblib
    import os
    import os.path as osp
    from tqdm import tqdm
    from scipy.spatial.transform import Rotation as sR


    device = 'cuda:0'
    batch_size = 1
    smpl = SMPL(hmr_config.SMPL_MODEL_DIR,
                batch_size=batch_size,
                create_transl=False).to(device)
    smpl_pred_output = smpl(betas=None, body_pose=None, global_orient=None)

    smplx = SMPLX('/home/groups/syyeung/wangkua1/smplx_models/smplx',
                  batch_size=1,
                  create_transl=False).to(device)
    smplx_pred_output = smplx(betas=None, body_pose=None, global_orient=None)

    # smpl_renderer = Renderer(focal_length=20000, # larger value render larger things.
    #                          img_width=500,
    #                          img_height=500,
    #                          faces=smpl.faces)
    smplx_renderer = Renderer(focal_length=3000,
                              img_width=1080,
                              img_height=1920,
                              faces=smplx.faces)

    def f_render(pred, renderer, C=None, R=[0, 0, 0]):
        points3d = pred.vertices
        if C is None:
            C = [0, 0, 100]
        # camera_translation = [0, 0, 100]
        # R = np.eye(3).astype('float32')
        # # R *= -1

        camera_translation = C  
        r = sR.from_euler('xyz', R, degrees=True) # (sagittal, )
        # [0, 0, 0] -> only foot
        # [90, 0, 0] -> back shot (upright)
        # [90, 0, 90] -> back shot (horizontal)
        # [90, 0, 180] -> back shot (upside down)
        # [90, 180, 0] -> nothing
        # [180, 0, 0] -> top of head
        # [270, 0, 0] -> nothing ... 
        R = r.as_matrix().astype('float32')
        print(">>> R")
        print(R)
        
        transformed_points3d = apply_extrinsics(
            points3d,
            rotation=torch.tensor(R).expand(batch_size, -1, -1).to(device),
            translation=torch.tensor(camera_translation)[None].expand(
                batch_size, -1).to(device))
        im = renderer(transformed_points3d[0].detach().cpu().numpy(),
                      np.zeros_like(camera_translation).astype('float32'),
                      np.zeros((1920, 1080, 3)),
                      return_camera=False)
        im = 255 * im
        im = im.astype('uint8')
        if im.max() > 0:
            print("something...")
        return im

    # im = f_render(smpl_pred_output, smpl_renderer)
    # cv2.imwrite(f'_smpl_viz.png', im)

    im = f_render(smplx_pred_output, smplx_renderer)
    cv2.imwrite(f'_smplx_viz2.png', im)

    # Data
    dir_path = '/home/groups/syyeung/wangkua1/data/mymocap/sub1'
    # tennis_forehand_stageii.pkl  tennis_serve_stageii.pkl
    fname = 'tennis_serve_stageii.pkl'
    # fname = 'tennis_forehand_stageii.pkl'
    data = joblib.load(osp.join(dir_path, fname))
    data['trans'] = data['trans'] - data['trans'][0]
    data['trans'][..., 2] += 0.9
    
    def to_tensor(npdata):
        return torch.tensor(npdata).float().to(device)
    # ipdb.set_trace()
    # smplx.betas

    # for ridx, R in enumerate(product([0, 90, 180, 270], repeat=3)):
    #     r0, r1, r2 = R
    #     dname = f'__quick_viz_{ridx}_{r0}_{r1}_{r2}'
    #     os.makedirs(dname, exist_ok=True)
    #     for idx in tqdm(range(0, 1200, 120)):
    #         preds = smplx(
    #             betas=None,  # to_tensor(data['betas'])[:10][None],
    #             # global_orient=torch.zeros_like(to_tensor(data['fullpose'][idx:idx+1, :3])),
    #             global_orient=to_tensor(data['fullpose'][idx:idx+1, :3]),
    #             body_pose=to_tensor(data['fullpose'][idx:idx+1, 3:(21 + 1) * 3]),
    #             pose2rot=True)
    #         preds.vertices = preds.vertices + to_tensor(data['trans'][idx:idx+1])
    #         # im = f_render(preds, smplx_renderer, C=data['trans'][0]) # I stood very close to the camera at frame 0
    #         im = f_render(preds, smplx_renderer, C=[0, 0, 20], R=R)
    #         cv2.imwrite(osp.join(dname, f'{idx:06d}.png'), im)

    # for ridx, R_diff in enumerate(product([-40, -20, 0, 20, 40], repeat=2)):
    #     r0, r1 = R_diff
    #     R_base = [90, 180, 0]
    #     R = [R_base[0] + r0, R_base[1] + r1, 0]
    #     # ridx = 0
    #     r0, r1, r2 = R
    #     dname = f'_quick_viz_{ridx}_{r0}_{r1}'
    #     os.makedirs(dname, exist_ok=True)
    #     for idx in tqdm(range(0, 1200, 120)):
    #         preds = smplx(
    #             betas=None,  # to_tensor(data['betas'])[:10][None],
    #             # global_orient=torch.zeros_like(to_tensor(data['fullpose'][idx:idx+1, :3])),
    #             global_orient=to_tensor(data['fullpose'][idx:idx+1, :3]),
    #             body_pose=to_tensor(data['fullpose'][idx:idx+1, 3:(21 + 1) * 3]),
    #             pose2rot=True)
    #         preds.vertices = preds.vertices + to_tensor(data['trans'][idx:idx+1])
    #         # im = f_render(preds, smplx_renderer, C=data['trans'][0]) # I stood very close to the camera at frame 0
    #         im = f_render(preds, smplx_renderer, C=[0, 1.5, 5], R=R)
    #         cv2.imwrite(osp.join(dname, f'{idx:06d}.png'), im)

    R = [90, 160, 0]
    dname = f'_quick_viz_kp'
    os.makedirs(dname, exist_ok=True)
    for idx in tqdm(range(0, 1200, 120)):
        preds = smplx(
            betas=None,  # to_tensor(data['betas'])[:10][None],
            global_orient=to_tensor(data['fullpose'][idx:idx+1, :3]),
            body_pose=to_tensor(data['fullpose'][idx:idx+1, 3:(21 + 1) * 3]),
            pose2rot=True)
        preds.vertices = preds.vertices + to_tensor(data['trans'][idx:idx+1])
        im = f_render(preds, smplx_renderer, C=[0, 1.5, 5], R=R)
        cv2.imwrite(osp.join(dname, f'{idx:06d}.png'), im)





    # SMPLX stuff...

    # def append_two_null_joints_to_pose(pose):
    #     return torch.cat(
    #         [pose, torch.zeros((pose.size(0), 6)).to(pose.device)], 1)

    # preds = smpl(
    #     betas=to_tensor(data['betas'])[:10][None],
    #     global_orient=torch.zeros_like(to_tensor(data['fullpose'][:1, :3])),
    #     # global_orient=to_tensor(data['fullpose'][:1, :3]),
    #     body_pose=append_two_null_joints_to_pose(
    #         to_tensor(data['fullpose'][:1, 3:(21 + 1) * 3])),
    #     pose2rot=True)

    # im = f_render(preds, smpl_renderer)
    # cv2.imwrite(f'_smpl_viz4.png', im)