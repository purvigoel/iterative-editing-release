import os
import torch
from torch.autograd import Variable
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from VIBE.lib.core.config import VIBE_DATA_DIR
from VIBE.lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from VIBE.lib.models.spin import Regressor, projection
from VIBE.lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS, SMPL_TO_J14

import mdm.utils.rotation_conversions as geometry

from gthmr.lib.models.decoder import MDMDecoder
from mdm.data_loaders.convert_motion_representations import HumamnMlConverter
from .vibe_component import TemporalEncoder

dpm_solver_args = {
            'algorithm_type': 'dpmsolver++',
            'method': 'multistep',
            'order': 3,
            'steps': 20,
        }


class Regress2MDM(nn.Module):

    def __init__(self,
                 decoder_model_path,
                 smpl_mean_params=SMPL_MEAN_PARAMS,
                 predicts_root_traj=False,
                 predict_rigid_trans=False,
                 weight_extra_loss=0.,
                 init_pose_gain=0.01,
                 normalize_time_only=True,
                 use_dpm_solver=True,
                 **kwargs):
        super(Regress2MDM, self).__init__()
        self.predict_rigid_trans = predict_rigid_trans
        self.weight_extra_loss = weight_extra_loss
        self.normalize_time_only = normalize_time_only
        self.kwargs = kwargs
        assert kwargs['n_joints'] == 14  #  the only supported mode.

        npose = 263
        # extra_dim = 3 if predicts_root_traj else 0
        self.fc1 = nn.Linear(512 * 4, 1024)
        # self.fc1 = nn.Linear(512 * 4 + npose + 3 + extra_dim, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.nonlin = nn.ReLU()
        self.norm = nn.BatchNorm1d(1024)

        self.decpose = nn.Linear(1024, npose, bias=False)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_normal_(self.decpose.weight, gain=init_pose_gain)
        # nn.init.xavier_uniform_(self.decpose.weight, gain=init_pose_gain)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        if self.predict_rigid_trans:
            self.decrigid = nn.Linear(1024, 10)
            nn.init.xavier_uniform_(self.decrigid.weight, gain=0.01)

        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)

        if predicts_root_traj:
            self.decrt = nn.Linear(1024, 3)
            nn.init.xavier_uniform_(self.decrt.weight, gain=0.01)
            init_rt = torch.zeros((1, 3))
            self.register_buffer('init_rt', init_rt)
        self.predicts_root_traj = predicts_root_traj

        self.decoder = MDMDecoder(decoder_model_path,
                                  use_dpm_solver=use_dpm_solver,
                                  dpm_solver_args=dpm_solver_args)

        self.converter = HumamnMlConverter()

    def mdm_output_to_joints(self, pred_pose6d):
        return self.converter.humanml3d_to_joint_positions(pred_pose6d,
                                                           no_root=True)

    def forward(self,
                x,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                init_rt=None,
                n_iter=3,
                J_regressor=None,
                dont_use_decoder=False):
        assert dont_use_decoder == False
        del J_regressor  # can't use this without SMPL

        # x.shape (bs ,T, D)
        sample_size, seqlen, feat_dim = x.shape
        x = x.reshape(-1, x.size(-1))  # (sample_size*seqlen, feat_dim)
        batch_size = x.shape[0]  # the new batch_size= sample_size*seqlen

        xc = self.fc1(x)
        # xc = self.drop1(xc)
        xc = self.nonlin(xc)
        xc = self.norm(xc)

        xc = self.fc2(xc)
        # xc = self.drop2(xc)
        xc = self.nonlin(xc)
        xc = self.norm(xc)
        
        pred_pose_inp = self.decpose(xc)
        pred_cam = self.deccam(xc)

        # put regressor outputs through MDM
        pred_pose_inp = pred_pose_inp.reshape(sample_size, seqlen, 263, 1)
        pred_pose_inp = pred_pose_inp.permute(0, 2, 3, 1)  # -> (N, J, 6, T)

        # Normalize input to MDM
        def _normalize(x, timeonly=True):
            if timeonly:
                x = (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)
            else:
                x = (x - x.mean([1, 2, 3], keepdims=True)) / x.std(
                    [1, 2, 3], keepdims=True)
            return x

        # pred_pose_inp = _normalize(pred_pose_inp, self.normalize_time_only)

        pred_pose_inp += torch.randn_like(pred_pose_inp) * 0.08
        pred_pose6d = self.decoder.decode(pred_pose_inp)
        joints3d_smpl22 = self.mdm_output_to_joints(
            pred_pose6d)  # (N, 22, 3, T)

        if self.predict_rigid_trans:
            pred_rigid = self.decrigid(xc)
            pred_rigid = pred_rigid.reshape(sample_size, seqlen, 10)
            # Average over frames since there's only 1 transform per sequence.
            pred_rigid = pred_rigid.mean(1, keepdims=True)
            pred_rigid_transl = pred_rigid[..., :3]
            pred_rigid_rot6d = pred_rigid[..., 3:9]  # (B, 1, 6)
            pred_rigid_scale = torch.ones_like(pred_rigid[..., 9:])  # (B, 1, 1)
            # pred_rigid_scale = pred_rigid[..., 9:]  # (B, 1, 1)
            pred_rigid_rotmat = geometry.rotation_6d_to_matrix(
                pred_rigid_rot6d)  # (B, 1, 3, 3)

            # Apply rigid transform to 3D-xyz
            joints3d_smpl22 = joints3d_smpl22.permute(0, 3, 1, 2).reshape(
                -1, 3)  # (N * T * 22, 3)
            pred_rigid_rotmat_ = pred_rigid_rotmat.unsqueeze(2).repeat(
                1, seqlen, 22, 1, 1).reshape(-1, 3, 3)
            pred_rigid_transl_ = pred_rigid_transl.unsqueeze(2).repeat(
                1, seqlen, 22, 1).reshape(-1, 3)
            pred_rigid_scale_ = pred_rigid_scale.unsqueeze(2).repeat(
                1, seqlen, 22, 1).reshape(-1, 1)

            # Rotate
            joints3d_smpl22 = torch.bmm(
                pred_rigid_rotmat_, joints3d_smpl22.unsqueeze(2)).squeeze(2)

            # Translate
            joints3d_smpl22 += pred_rigid_transl_

            # Scale
            joints3d_smpl22 *= pred_rigid_scale_

            # Reshape back
            joints3d_smpl22 = joints3d_smpl22.reshape(sample_size, seqlen, 22,
                                                      3)
            joints3d_smpl22 = joints3d_smpl22.permute(0, 2, 3, 1)

        # Postprocessing
        pred_joints = joints3d_smpl22[:, SMPL_TO_J14]

        # Reshape
        joints3d_smpl22 = joints3d_smpl22.permute(0, 3, 1,
                                                  2).reshape(-1, 22, 3)
        pred_joints = pred_joints.permute(0, 3, 1, 2).reshape(-1, 14, 3)

        # Regularize
        extra_loss = 0.
        if self.weight_extra_loss > 0:
            extra_loss = (pred_pose_inp**2).mean()
            extra_loss = self.weight_extra_loss * extra_loss
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        output = [{
            'cam': pred_cam,
            # 'theta': torch.cat([pred_cam, pose, pred_shape], dim=1),
            # 'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'smpl_joint': joints3d_smpl22,
            'extra_loss': extra_loss,
        }]

        return output


class MyRegressor(nn.Module):

    def __init__(self,
                 decoder_model_path,
                 smpl_mean_params=SMPL_MEAN_PARAMS,
                 predicts_root_traj=False,
                 use_dpm_solver=True,
                 **kwargs):
        super(MyRegressor, self).__init__()
        self.kwargs = kwargs
        npose = 24 * 6
        extra_dim = 3 if predicts_root_traj else 0
        self.fc1 = nn.Linear(512 * 4 + npose + 13 + extra_dim, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(
            mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        if predicts_root_traj:
            self.decrt = nn.Linear(1024, 3)
            nn.init.xavier_uniform_(self.decrt.weight, gain=0.01)
            init_rt = torch.zeros((1, 3))
            self.register_buffer('init_rt', init_rt)
        self.predicts_root_traj = predicts_root_traj

        self.decoder = MDMDecoder(decoder_model_path,
                                  use_dpm_solver=use_dpm_solver,
                                  dpm_solver_args=dpm_solver_args)

    def forward(self,
                x,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                init_rt=None,
                n_iter=3,
                J_regressor=None,
                dont_use_decoder=False):
        # x.shape (bs ,T, D)
        sample_size, seqlen, feat_dim = x.shape
        x = x.reshape(-1, x.size(-1))  # (sample_size*seqlen, feat_dim)
        batch_size = x.shape[0]  # the new batch_size= sample_size*seqlen

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        if self.predicts_root_traj:
            if init_rt is None:
                init_rt = self.init_rt.expand(batch_size, -1)
            pred_pose = init_pose
            pred_shape = init_shape
            pred_cam = init_cam
            pred_rt = init_rt
            for i in range(n_iter):
                xc = torch.cat([x, pred_pose, pred_shape, pred_cam, pred_rt],
                               1)
                xc = self.fc1(xc)
                xc = self.drop1(xc)
                xc = self.fc2(xc)
                xc = self.drop2(xc)
                pred_pose = self.decpose(xc) + pred_pose
                pred_shape = self.decshape(xc) + pred_shape
                pred_cam = self.deccam(xc) + pred_cam
                pred_rt = self.decrt(xc) + pred_rt

        else:  # No RT
            pred_pose = init_pose
            pred_shape = init_shape
            pred_cam = init_cam
            for i in range(n_iter):
                xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
                xc = self.fc1(xc)
                xc = self.drop1(xc)
                xc = self.fc2(xc)
                xc = self.drop2(xc)
                pred_pose = self.decpose(xc) + pred_pose
                pred_shape = self.decshape(xc) + pred_shape
                pred_cam = self.deccam(xc) + pred_cam

        if dont_use_decoder:
            pred_pose6d = pred_pose
            pred_pose6d_decoder_output = torch.clone(
                pred_pose6d.detach()) if not self.training else None
            root_traj = None
            pred_rotmat = geometry.rotation_6d_to_matrix(
                pred_pose6d.reshape(-1, 24, 6)).view(batch_size, 24, 3, 3)
            # pred_rotmat = rot6d_to_rotmat(pred_pose6d).view(batch_size, 24, 3, 3)
        else:
            # put regressor outputs through MDM
            pred_pose6d = pred_pose.reshape(sample_size, seqlen, 24, 6)
            # Prepend 0's as global traj
            root_traj = torch.zeros((sample_size, seqlen, 1, 6),
                                    device=pred_pose6d.device)
            if self.predicts_root_traj:
                root_traj[:, :, :, :3] = pred_rt.reshape(
                    sample_size, seqlen, 1, 3)

            pred_pose6d = torch.cat([pred_pose6d, root_traj], 2)
            pred_pose6d = pred_pose6d.permute(0, 2, 3, 1)  # -> (N, J, 6, T)
            pred_pose6d = self.decoder.decode(pred_pose6d)
            pred_pose6d_decoder_output = torch.clone(pred_pose6d.detach(
            )) if not self.training else None  ## for eval shape (N,25,6,T)
            # reshaping
            pred_pose6d = pred_pose6d.permute(0, 3, 1, 2)  # -> (N, T, J, 6)
            root_traj = pred_pose6d[:, :, -1:, :3]  # (N, T, 1, 3) 
            # pred_pose6d = pred_pose6d[:, :, :-1].reshape(
            # batch_size, -1)  # -> (N * T, J * 6)
            pred_pose6d = pred_pose6d[:, :, :-1].reshape(
                batch_size, 24, 6)  # -> (N * T, J, 6)
            # Note: the function below is different from rot6d_to_rotmat
            pred_rotmat = geometry.rotation_6d_to_matrix(pred_pose6d)

        pred_output = self.smpl(betas=pred_shape,
                                body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                                pose2rot=False)

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        smpl_joints_w_trans = pred_output.smpl_joints + root_traj.reshape(-1, 1, 3)
        # ipdb.set_trace()

        if self.kwargs['n_joints'] == 14:
            pred_joints = pred_output.smpl_joints[:, SMPL_TO_J14]

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(
                pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3,
                                                                 3)).reshape(
                                                                     -1, 72)

        output = [{
            'theta': torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'smpl_output': pred_output,
            'smpl_joints_w_trans': smpl_joints_w_trans,
            'rotmat': pred_rotmat,
            'root_traj': root_traj,
            'decoder_out': pred_pose6d_decoder_output,  # raw decoder output
            'regressor_pred_pose': pred_pose,  # pre-decoder pose 
            'regressor_pred_shape': pred_shape,  # pre-decoder shape 
            'regressor_pred_cam': pred_cam,  # pre-decoder camera
        }]

        return output


class Model0(nn.Module):

    def __init__(self,
                 args,
                 seqlen,
                 temporal_type='gru',
                 batch_size=64,
                 n_layers=1,
                 hidden_size=2048,
                 add_linear=False,
                 bidirectional=False,
                 use_residual=True,
                 pretrained=osp.join(VIBE_DATA_DIR,
                                     'spin_model_checkpoint.pth.tar'),
                 decoder_model_path='mdm/save/unconstrained/model000450000.pt',
                 n_joints=49):

        super(Model0, self).__init__()
        self.args = args
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.n_joints = n_joints

        if temporal_type == 'gru':
            self.encoder = TemporalEncoder(
                n_layers=n_layers,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                add_linear=add_linear,
                use_residual=use_residual,
            )
        elif temporal_type == 'none':
            self.encoder = lambda x: x

        # regressor can predict cam, pose and shape params in an iterative way

        self.my_regressor = eval(args.REGRESSOR_CLASS)(
            decoder_model_path,
            predicts_root_traj=args.PREDICTS_ROOT_TRAJ,
            predict_rigid_trans=args.PREDICT_RIGID,
            weight_extra_loss=args.WEIGHT_EXTRA_LOSS,
            init_pose_gain=args.INIT_POSE_GAIN,
            normalize_time_only=args.NORMALIZE_TIME_ONLY,
            n_joints=n_joints,
            use_dpm_solver=args.USE_DPM_SOLVER)

        # if pretrained and os.path.isfile(pretrained):
        #     pretrained_dict = torch.load(pretrained)['model']

        #     self.my_regressor.load_state_dict(pretrained_dict, strict=False)
        #     print(f'=> loaded pretrained regressor from \'{pretrained}\'')

    def forward(self, input, J_regressor=None):
        smpl_output_new = self.new_forward(input, J_regressor)
        return smpl_output_new

    def new_forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen, feat_dim = input.shape
        feature = self.encoder(input)

        smpl_output = self.my_regressor(
            feature,
            J_regressor=J_regressor,
            dont_use_decoder=self.args.DONT_USE_DECODER)
        for s in smpl_output:
            if 'theta' in s:
                s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
                s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['batch_size'] = batch_size
            s['seqlen'] = seqlen
        return smpl_output
