import torch
import torch.nn as nn

from VIBE.lib.utils.geometry import batch_rodrigues
import ipdb

class VIBELoss(nn.Module):

    def __init__(
        self,
        e_loss_weight=60.,
        e_3d_loss_weight=30.,
        e_3d_vel_loss_weight=30.,
        e_pose_loss_weight=1.,
        e_shape_loss_weight=0.001,
        device='cuda',
        n_joints=49,
    ):
        super(VIBELoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_3d_vel_loss_weight = e_3d_vel_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.n_joints = n_joints

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

    def forward(
        self,
        generator_outputs,
        data_2d,
        data_3d,
    ):
        # to reduce time dimension
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1], ) + x.shape[2:]) 
        # flatten for weight vectors
        flatten = lambda x: x.reshape(-1)

        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
        else:
            sample_2d_count = 0
            real_2d = data_3d['kp_2d']
        
        batch_size, seqlen = data_3d['kp_3d'].shape[:2]

        # Collect data
        real_2d = reduce(real_2d)
        real_3d = reduce(data_3d['kp_3d'])
        
        w_3d = data_3d['w_3d'].type(torch.bool)
        w_smpl = data_3d['w_smpl'].type(torch.bool)
        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)
        real_3d = real_3d[w_3d]

        if self.n_joints == 14:
            real_2d = real_2d[:, 25:39]
            real_3d = real_3d[:, 25:39]

        # Collect Preds
        preds = generator_outputs[-1]
        pred_j3d = preds['kp_3d'][sample_2d_count:]
        if len(pred_j3d.shape) == 4:
            pred_j3d = reduce(pred_j3d)
        pred_j2d = preds['kp_2d']
        if len(pred_j2d.shape) == 4:
            pred_j2d = reduce(pred_j2d)
        pred_j3d = pred_j3d[w_3d]
        
        # <======== Generator Loss
        loss_kp_2d = self.keypoint_loss(
            pred_j2d, real_2d, openpose_weight=1.,
            gt_weight=1.) * self.e_loss_weight

        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight



        loss_dict = {
            'loss_kp_2d': loss_kp_2d,
            'loss_kp_3d': loss_kp_3d,
        }
        
        if self.e_3d_vel_loss_weight > 0:
            loss_kp_vel_3d = self.keypoint_3d_vel_loss(pred_j3d, real_3d, seqlen=seqlen)
            loss_kp_vel_3d = loss_kp_vel_3d * self.e_3d_vel_loss_weight
            loss_dict['loss_kp_3d_vel'] = loss_kp_vel_3d

        if self.e_pose_loss_weight > 0:

            data_3d_theta = reduce(data_3d['theta'])
            pred_theta = preds['theta'][sample_2d_count:]
            if len(pred_theta.shape) == 3:
                pred_theta = reduce(pred_theta)
            pred_theta = pred_theta[w_smpl]
            data_3d_theta = data_3d_theta[w_smpl]
            
            real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
            real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]


            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape,
                                                     real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose

        gen_loss = torch.stack(list(loss_dict.values())).sum()

        return gen_loss, loss_dict

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d,
                      openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(
            pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        if self.n_joints == 49:
            pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
            gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]

        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] +
                         gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] +
                           pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
            # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            return self.criterion_keypoints(pred_keypoints_3d,
                                            gt_keypoints_3d).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    
    def keypoint_3d_vel_loss(self, pred_keypoints_3d, gt_keypoints_3d, seqlen=16):
        """
        Compute 3D keypoint velocity loss.
        This is an attempt to solve the "static" segment prediction issue.
        """
        if self.n_joints == 49:
            pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
            gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]

        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] +
                         gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] +
                           pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

            # Reshape
            gt_keypoints_3d = gt_keypoints_3d.reshape(-1, seqlen, 14, 3)
            pred_keypoints_3d = pred_keypoints_3d.reshape(-1, seqlen, 14, 3)

            # Diff
            gt_keypoints_3d_diff = gt_keypoints_3d[:, 1:] - gt_keypoints_3d[:, :-1]
            pred_keypoints_3d_diff = pred_keypoints_3d[:, 1:] - pred_keypoints_3d[:, :-1]

            return self.criterion_keypoints(pred_keypoints_3d_diff,
                                            gt_keypoints_3d_diff).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)


    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(
            -1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(
            -1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid,
                                                 gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid,
                                                  gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas


def batch_smooth_pose_loss(pred_theta):
    pose = pred_theta[:, :, 3:75]
    pose_diff = pose[:, 1:, :] - pose[:, :-1, :]
    return torch.mean(pose_diff).abs()


def batch_smooth_shape_loss(pred_theta):
    shape = pred_theta[:, :, 75:]
    shape_diff = shape[:, 1:, :] - shape[:, :-1, :]
    return torch.mean(shape_diff).abs()
