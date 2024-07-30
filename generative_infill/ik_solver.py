import torch
import numpy as np
import pose_mask
import torch.optim as optim
from torch.autograd import Variable

class IKSolver:
    def __init__(self):
        self.here = True

    def cosine_rule(self, a, b, c):
        norm_ab = (c-a)/torch.norm(c - a)
        norm_ba = (b-a)/ torch.norm(b - a)

        dotted = torch.dot(norm_ab, norm_ba)
        dotted_clamped = self.clamp(dotted, -1, 1)
        return torch.acos(dotted_clamped)
    
    def clamp(self, a, lo, hi):
        if lo >= a:
            a = lo
        elif hi <= a:
            a = hi
        return a

    def quat_angle_axis(self, sample, direc, model):
        root_orient = sample:, 0:6, ...].clone().squeeze(-1).squeeze(-1)
        root_orient_aa = geometry.rotation_6d_to_aa(root_orient)
        root_orient_mat = batch_rodrigues(root_orient_aa).view(-1, 3, 3)
        diff = torch.matmul(root_orient_mat[j].float(), direc)

        return diff

    # two-bone analytical solution
    def forward(self, sample, model):
        sample = sample[...,0].unsqueeze(-1)
        current_pos = model.forward_kinematics(sample, None)["kp_45_joints"][:, :24].reshape(1, 24,3)
        # hip
        a = current_pos[0,2,:]
        b = current_pos[0,5,:]
        c = current_pos[0,8,:]
        t = c + torch.tensor([0.0 , 0.25, 0.0]).to(c.device)
        eps = 0.01
        lab = torch.norm(b - a)
        lcb = torch.norm(b - c)
        lat = self.clamp( torch.norm(t - a), eps, lab + lcb - eps)

        ac_ab_0 = self.cosine_rule(a, b, c)
        ba_bc_0 = self.cosine_rule(b, c, a)
        ac_at_0 = self.cosine_rule(a, t, c)

        ac_ab_1 = torch.acos( self.clamp( (lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat) , -1, 1))
        ba_bc_1 = torch.acos( self.clamp( (lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb) , -1, 1))

        axis0 = torch.cross(c - a, b - a) 
        axis1 = torch.cross(c - a, t - a) 

        axis0 = axis0 / torch.norm(axis0)
        axis1 = axis1 / torch.norm(axis1)

        # axis0 
        2/0

        r0 = quat_angle_axis( ac_ab_1 - ac_ab_0, quat_mul( quat_inv(a_gr), axis0))
        r1 = quat_angle_axis( ba_bc_1 - ba_bc_0, quat_mul(quat_inv(b_gr), axis0))
        r2 = quat_angle_axis( ac_at_0 - quat_mul(quat_inv(a_gr), axis1))

        a_lr = quat_mul(a_lr, quat_mul(r0, r2))
        b_lr = quat_mul(b_lr, r1)

    def forward_optim(self, sample, model):
        sample = sample[..., 0].unsqueeze(-1)
        mask = torch.zeros(sample.shape).to(sample.device) #sample.shape).to(sample.device)
        mask[ :, pose_mask.right_leg_angles,:,  : ] = 1
        with torch.enable_grad():
            angles = Variable(sample, requires_grad = True)
            opt = optim.Adam([angles], lr=1e-2)

            target_joint_locations = model.forward_kinematics(sample, None)["kp_45_joints"][:, :24].reshape(1, 24,3)

            target = target_joint_locations[:, 8, :] + torch.tensor([0.0 , 0.2, 0.0]).unsqueeze(0).to(target_joint_locations.device)

            for itr in range(100):
                pred = model.forward_kinematics(angles, None)
                pred_joints = pred["kp_45_joints"][:, :24].reshape(1, 24,3) 
                
                loss = torch.norm(target - pred_joints[:, 8, :])
                opt.zero_grad()
                loss.backward()
                angles.grad *= mask
                opt.step() 
                
        np.save("ik_test.npy", pred_joints.detach().cpu().unsqueeze(0).numpy())
        return
