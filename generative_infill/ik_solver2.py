import torch
import numpy as np
import pose_mask
import torch.optim as optim
from torch.autograd import Variable
from mdm.utils.rotation_conversions import axis_angle_to_quaternion,axis_angle_to_6d,quaternion_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion, rotation_6d_to_aa
from VIBE.lib.utils.geometry import batch_rodrigues
import generative_infill.reloader as reloader

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
            a = torch.tensor(lo).to(a.device)
        elif hi <= a:
            a = torch.tensor(hi).to(a.device)
        return a

    '''
    def quat_angle_axis(self, sample, direc, model):
        root_orient = sample:, 0:6, ...].clone().squeeze(-1).squeeze(-1)
        root_orient_aa = geometry.rotation_6d_to_aa(root_orient)
        root_orient_mat = batch_rodrigues(root_orient_aa).view(-1, 3, 3)
        diff = torch.matmul(root_orient_mat[j].float(), direc)

        return diff
    '''

    def quat_mult(self, A, B):
        w0, x0, y0, z0 = A
        w1, x1, y1, z1 = B
        return torch.tensor([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]).to(A.device)


    def to_matrix(self, angle, axis0):
        r0_aa = torch.tensor([angle, axis0[0], axis0[1], axis0[2]]).to(axis0.device)
        r0_aa = r0_aa[1:] * r0_aa[0]
        r0_aa = r0_aa.unsqueeze(0)
        r0 = axis_angle_to_matrix(r0_aa)
        return r0

    def to_matrix2(self, angle, axis0):
        axis0 = axis0.squeeze(0)
        r0_aa = torch.tensor([angle, axis0[0], axis0[1], axis0[2]]).to(axis0.device)
        r0_aa = r0_aa[1:] * r0_aa[0]
        r0_aa = r0_aa.unsqueeze(0)
        r0 = axis_angle_to_matrix(r0_aa)
        return r0

    def vector_length(self, v):
        return torch.sqrt(torch.sum(v ** 2))

    def to_quat(self, mag, axis):
        quat = axis_angle_to_quaternion(mag * axis)
        return quat

    def reset_target(self, hip, leg_length, target):
        c2 = leg_length
        b2 = hip[1] - target[1]
        a2 = torch.sqrt( c2 * c2 - b2 * b2)
        if b2 * b2 > c2 * c2:
            a2 = c2

        root_projection = hip.clone()
        root_projection[1] = target[1]

        f_r = target - root_projection
        f_r = f_r / torch.linalg.norm(f_r)
        target = root_projection + f_r * a2
        
        return target

    def forward_quat(self, sample, model, frame, current_pos, current_trans, target, kin_chain=[1,4,7], rewrite_target=False,pos_ref=[]):
        sample = sample[...,frame].unsqueeze(-1).float()
        sample = sample.to(current_pos.device).float()
        target = target.float()

        # hip
        a = current_pos[kin_chain[0],:] - current_trans
        b = current_pos[kin_chain[1],:] - current_trans
        c = current_pos[kin_chain[2],:] - current_trans
        t = target - current_trans

        eps = 0.01
        lab = torch.norm(b - a)
        lcb = torch.norm(b - c)
        lat = self.clamp( torch.norm(t - a), eps, lab + lcb - eps)
        if torch.norm(t-a) > (lab+lcb-eps) and rewrite_target:
            t = self.reset_target(a, lab + lcb - eps, t)
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

        root_rotation = rotation_6d_to_matrix(sample[:, 0:6, :, :].squeeze())
        hip_rotation = rotation_6d_to_matrix(sample[:, kin_chain[0]*6: kin_chain[0]*6+6, :, :].squeeze())
        knee_rotation = rotation_6d_to_matrix(sample[:, kin_chain[1]*6: kin_chain[1]*6+6, :, :].squeeze())
        hip_global_to_local = torch.inverse(root_rotation)
        knee_global_to_local = torch.inverse(torch.matmul(root_rotation, hip_rotation))
        
        r0 = self.to_quat(ac_ab_1 - ac_ab_0, torch.matmul(hip_global_to_local, axis0))
        r1 = self.to_quat(ba_bc_1 - ba_bc_0, torch.matmul(knee_global_to_local, axis0))
        r2 = self.to_quat(ac_at_0,  torch.matmul(hip_global_to_local,axis1))

        hip_global_to_local_quat = matrix_to_quaternion(hip_rotation)
        a_lr = axis_angle_to_6d(quaternion_to_axis_angle(self.quat_mult(hip_global_to_local_quat, self.quat_mult(r0,r2))))

        knee_global_to_local_quat = matrix_to_quaternion(knee_rotation)
        b_lr = axis_angle_to_6d(quaternion_to_axis_angle(self.quat_mult(knee_global_to_local_quat, r1))) 
        
        hip_rot6d = a_lr
        sample[:, kin_chain[0]*6: kin_chain[0]*6+6, :, :] = hip_rot6d.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        knee_rot6d = b_lr
        sample[:, kin_chain[1]*6: kin_chain[1]*6+6, :, :] = knee_rot6d.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return sample

    def foot_floor_align(self, sample, model, frame, current_pos, current_trans, target, kin_chain=[1,4,7]):
        sample = sample[...,frame].unsqueeze(-1)
        sample = sample.to(current_pos.device)

        # ankle
        a = current_pos[kin_chain[1],:] - current_trans
        b = current_pos[kin_chain[2],:] - current_trans
        c = current_pos[kin_chain[3],:] - current_trans
        t = target - current_trans

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

        root_rotation = rotation_6d_to_matrix(sample[:, 0:6, :, :].squeeze())
        hip_rotation = rotation_6d_to_matrix(sample[:, kin_chain[0]*6: kin_chain[0]*6+6, :, :].squeeze())
        knee_rotation = rotation_6d_to_matrix(sample[:, kin_chain[1]*6: kin_chain[1]*6+6, :, :].squeeze())
        ankle_rotation = rotation_6d_to_matrix(sample[:, kin_chain[2]*6: kin_chain[2]*6+6, :, :].squeeze())

        hip_global_to_local = torch.inverse(root_rotation)
        knee_global_to_local = torch.inverse(torch.matmul(root_rotation, hip_rotation))
        ankle_global_to_local = torch.inverse(torch.matmul(torch.matmul(root_rotation, hip_rotation), ankle_rotation))

        r0 = self.to_quat(ac_ab_1 - ac_ab_0, torch.matmul(hip_global_to_local, axis0))
        r1 = self.to_quat(ba_bc_1 - ba_bc_0, torch.matmul(knee_global_to_local, axis0))
        r2 = self.to_quat(ac_at_0,  torch.matmul(hip_global_to_local,axis1))

        hip_global_to_local_quat = matrix_to_quaternion(knee_rotation)
        a_lr = axis_angle_to_6d(quaternion_to_axis_angle(self.quat_mult(hip_global_to_local_quat, self.quat_mult(r0,r2))))

        knee_global_to_local_quat = matrix_to_quaternion(ankle_rotation)
        b_lr = axis_angle_to_6d(quaternion_to_axis_angle(self.quat_mult(knee_global_to_local_quat, r1)))

        knee_rot6d = b_lr
        sample[:, kin_chain[2]*6: kin_chain[2]*6+6, :, :] = knee_rot6d.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return sample


    # two-bone analytical solution
    def forward(self, sample, model, frame, current_pos, current_trans, target, kin_chain=[1,4,7]):
        sample = sample[...,frame].unsqueeze(-1)
        sample = sample.to(current_pos.device)

        # hip
        a = current_pos[kin_chain[0],:] #- current_trans
        b = current_pos[kin_chain[1],:] #- current_trans
        c = current_pos[kin_chain[2],:] #- current_trans
        t = target #- current_trans
        
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
       
        r0 = self.to_matrix(ac_ab_1 - ac_ab_0, axis0)
        r1 = self.to_matrix(ba_bc_1 - ba_bc_0, axis0)
        r2 = self.to_matrix(ac_at_0, axis1)
    
        
        root_rotation = rotation_6d_to_matrix(sample[:, 0:6, :, :].squeeze().unsqueeze(0))
        hip_rotation = rotation_6d_to_matrix(sample[:, kin_chain[0]*6: kin_chain[0]*6+6, :, :].squeeze().unsqueeze(0))
        knee_rotation = rotation_6d_to_matrix(sample[:, kin_chain[1]*6: kin_chain[1]*6+6, :, :].squeeze().unsqueeze(0))

        # global rotation of hip
        hip_global_to_local = torch.inverse(root_rotation)

        # global rotation of knee
        knee_global_to_local = torch.inverse(torch.matmul(root_rotation, hip_rotation)) 
        
        r0 = self.to_matrix2(ac_ab_1 - ac_ab_0, torch.matmul(hip_global_to_local, axis0))
        r1 = self.to_matrix2(ba_bc_1 - ba_bc_0, torch.matmul(knee_global_to_local, axis0))
        r2 = self.to_matrix2(ac_at_0, torch.matmul(hip_global_to_local, axis1))

        a_lr = torch.matmul(hip_rotation, torch.matmul(r0,r2))
        b_lr = torch.matmul(knee_rotation, r1)
        print("hip rotation", hip_rotation)
        print("knee rotation", knee_rotation) 
        
        hip_rot6d = matrix_to_rotation_6d(a_lr)
        sample[:, kin_chain[0]*6: kin_chain[0]*6+6, :, :] = hip_rot6d.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        knee_rot6d = matrix_to_rotation_6d(b_lr)
        sample[:, kin_chain[1]*6: kin_chain[1]*6+6, :, :] = knee_rot6d.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        print("hip rotation", a_lr)
        print("knee rotation", b_lr)
        return sample

    def interpolate(self, og_motion, sample, lo_frame, lo_window, hi_frame, hi_window, side="right_foot", only_hi=False):
        to_infill = sample.clone()
        keyframe = sample.clone()

        if side == "right_foot":
            leg_angles = pose_mask.right_leg_angles
        elif side == "left_foot":
            leg_angles = pose_mask.leg_leg_angles
        elif side == "right_hand":
            leg_angles = pose_mask.right_arm_2joint_angles
        elif side == "left_hand":
            leg_angles = pose_mask.left_arm_2joint_angles

        if not only_hi and lo_frame >= 0:
            keyf = keyframe[:, leg_angles, :, lo_frame + 1]
            delta = keyf.clone() - og_motion[:,leg_angles,:,lo_frame + 1].clone()
            
            for fr in range(lo_window, lo_frame):
                a = (fr - lo_window) / (lo_frame - lo_window)
                og_motion[:, leg_angles, :, fr] = og_motion[:, leg_angles, :, fr] + (a) * (delta)
            to_infill[:, leg_angles, : , lo_frame:lo_window] = og_motion[:, leg_angles, : ,lo_frame:lo_window].clone()

        if hi_frame < sample.shape[-1]:
            keyf = keyframe[:, leg_angles, :, hi_frame - 1]
            delta = keyf.clone() - og_motion[:, leg_angles,:, hi_frame - 1].clone()
            for fr in range(hi_frame, hi_window):
                a = (fr - hi_frame) / (hi_window - hi_frame)
                og_motion[:, leg_angles, :, fr] = og_motion[:, leg_angles, :, fr] + (1 - a) * (delta)
        
            to_infill[:, leg_angles, : , hi_frame:hi_window] = og_motion[:, leg_angles, : , hi_frame:hi_window].clone()
        return to_infill

    def fk_shortcut(self, sample, model):
        current_pos_dic = model.forward_kinematics(sample, None)
        current_pos = current_pos_dic["kp_45_joints"][:, :22].reshape(sample.shape[-1], 22,3) + current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_trans = current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_pos = current_pos.to(sample.device)
        return current_pos

    def forward_fix_foot(self, sample, model):
        current_pos_dic = model.forward_kinematics(sample, None)
        current_pos = current_pos_dic["kp_45_joints"][:, :22].reshape(sample.shape[-1], 22,3) + current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_trans = current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_pos = current_pos.to(sample.device)
        
        c = current_pos[0, 8,:]
        t = c
        t = t.clone()

        for i in range(0, sample.shape[-1]):
            sample[..., i] = self.forward(sample, model, i, current_pos[i], current_trans[i], t, kin_chain=[2,5,8]).squeeze(-1)
        return sample

    def ws_mag_to_local(self, sample, magnitude, frame):
        root_orient = sample[:, 0:6, ..., frame].clone().squeeze()
        root_orient_aa = rotation_6d_to_aa(root_orient).unsqueeze(0)
        root_orient_mat = batch_rodrigues(root_orient_aa).view( 3, 3)

        tmp_magnitude = magnitude.clone()
        tmp_magnitude[1] = 0.0
        if(magnitude[1] == 0.0):
            oldnorm = np.linalg.norm(tmp_magnitude.cpu().numpy())
            direc = tmp_magnitude.float() / oldnorm
            diff = torch.matmul(root_orient_mat.float(), direc)
            diff = diff * oldnorm
            tmp_magnitude = diff
        tmp_magnitude[1] += magnitude[1]
        #print(tmp_magnitude)
        return tmp_magnitude

    def to_target(self, sample, model, frame, mag, side="right_foot", ws=True,rewrite_target=False):
        if side=="right_foot":
            kin_chain = [2,5,8]
        elif side == "left_foot":
            kin_chain = [1, 4, 7]
        elif side == "right_hand":
            kin_chain = [17, 19, 21]
        elif side == "left_hand":
            kin_chain = [16, 18, 20]
        mag = mag.to(sample.device)
        if ws:
            mag = self.ws_mag_to_local(sample, mag, frame)

        current_pos_dic = model.forward_kinematics(sample, None)
        current_pos = current_pos_dic["kp_45_joints"][:, :22].reshape(sample.shape[-1], 22,3) + current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_trans = current_pos_dic["pred_trans"].reshape(sample.shape[-1],3)
        
        t = current_pos[frame, kin_chain[-1],:]
        t = t.clone() + mag.to(t.device)
        sample[...,frame] = self.forward_quat(sample, model, frame, current_pos[frame], current_trans[frame], t, kin_chain=kin_chain, rewrite_target=rewrite_target,pos_ref=[current_pos,current_trans]).squeeze(-1)
        
        sample = reloader.reload(model, sample)
        return sample

    def fix_foot_slide_all(self, sample, model, frame, side="right", force=False):
        if side=="right":
            kin_chain = [2,5,8]
            ball = 3
            toe = 2
        else:
            kin_chain = [1, 4, 7]
            ball = 1
            toe = 0

        sample = reloader.reload(model, sample)
        old_sample = sample.clone()
        fc_pred = sample[0, 150:154, 0, :]
        foot_slide = []
        for f in range(0, fc_pred.shape[-1]):
            if (fc_pred[ball, f] > 0 and fc_pred[toe, f] > 0):
               foot_slide.append(f) 
        print(foot_slide)

        new_foot_slide = foot_slide
        for i in range(0, len(foot_slide) - 2):
            if foot_slide[i] + 2 in foot_slide and foot_slide[i] + 1 not in foot_slide:
                new_foot_slide.append(foot_slide[i] + 1)
        new_foot_slide.sort()
        foot_slide = new_foot_slide.copy()
        new_foot_slide = []
        for i in range(1, len(foot_slide)):
            if foot_slide[i] - 1 != foot_slide[i - 1]:
                new_foot_slide.append( foot_slide[i])

        for i in new_foot_slide:
            self.fix_foot_slide(sample, model, i, side, frame_force=True)
        sample = reloader.reload(model, sample)

        return sample

    def fix_foot_slide(self, sample, model, frame, side="right", force=False, all_fr=False, frame_force=False):
        # Given frame, fix the foot the position C
        # look back  frames. If they are on the ground, fix the foot to position C.
        # look forward  frames. If it is on the ground fix the foot to position C.
        # if not on the ground, lerp.
        if side=="right":
            kin_chain = [2,5,8]
            ball = 3
            toe = 2
        else:
            kin_chain = [1, 4, 7]
            ball = 1
            toe = 0

        sample = reloader.reload(model, sample)
        old_sample = sample.clone()
        fc_pred = sample[0, 150:154, 0, :]

        if frame_force:
            fc_pred[ball, frame] = 1
            fc_pred[toe, frame] = 1

        current_pos_dic = model.forward_kinematics(sample, None)
        current_pos = current_pos_dic["kp_45_joints"][:, :22].reshape(sample.shape[-1], 22,3) + current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_trans = current_pos_dic["pred_trans"].reshape(sample.shape[-1],3)

        t = current_pos[frame, kin_chain[-1],:]
        t = t.clone()

        f = frame
        lo_frame = f
        # look back
        while(f >= 0 and (force or (fc_pred[ball, f] > 0 and fc_pred[toe, f] > 0))):
            sample[..., f] = self.forward_quat(sample, model, f, current_pos[f], current_trans[f], t, kin_chain=kin_chain).squeeze(-1)
            f -= 1
        lo_frame = f

        f = frame
        hi_frame = f
        while(f < sample.shape[-1] and (force or (fc_pred[ball, f] > 0 and fc_pred[toe, f] > 0))):
            sample[..., f] = self.forward_quat(sample, model, f, current_pos[f], current_trans[f], t, kin_chain=kin_chain).squeeze(-1)
            f += 1
        hi_frame = f

        #blend
        L = 5

        lo_window = max(lo_frame - 5, -1)
        for i in range(lo_frame, lo_window, -1):
            if fc_pred[ball, i] > 0 and fc_pred[toe, i] > 0:
                lo_window = i
                break

        hi_window = min(hi_frame + 5, sample.shape[-1])
        for i in range(hi_frame, hi_window):
            if fc_pred[ball, i] > 0 and fc_pred[toe, i] > 0:
                hi_window = i
                break
        print(lo_window, lo_frame, hi_frame, hi_window)

        sample = self.interpolate(old_sample, sample, lo_frame, lo_window, hi_frame, hi_window)
        #assert(not torch.allclose(sample, old_sample))

        sample = reloader.reload(model, sample)
        return sample

    def fix_foot_slide_range(self, sample, model, start_frame, end_frame, side="right", force=False):
        # Given frame, fix the foot the position C
        # look back  frames. If they are on the ground, fix the foot to position C.
        # look forward  frames. If it is on the ground fix the foot to position C.
        # if not on the ground, lerp.

        if side=="right":
            kin_chain = [2,5,8]
            ball = 3
            toe = 2
        else:
            kin_chain = [1, 4, 7]
            ball = 1
            toe = 0

        sample = reloader.reload(model, sample)
        old_sample = sample.clone()
        fc_pred = sample[0, 150:154, 0, :]

        current_pos_dic = model.forward_kinematics(sample, None)
        current_pos = current_pos_dic["kp_45_joints"][:, :22].reshape(sample.shape[-1], 22,3) + current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_trans = current_pos_dic["pred_trans"].reshape(sample.shape[-1],3)

        t = current_pos[start_frame, kin_chain[-1],:]
        t = t.clone()

        lo_frame = start_frame
        hi_frame = end_frame
        # look back
        for f in range(start_frame, end_frame):
            print("foot slide range")
            sample[..., f] = self.forward_quat(sample, model, f, current_pos[f], current_trans[f], t, kin_chain=kin_chain).squeeze(-1)
        
        #blend
        L = 5
        hi_window = min(hi_frame + 5, sample.shape[-1])

        sample = self.interpolate(old_sample, sample, None, None, hi_frame, hi_window, only_hi = True)

        sample = reloader.reload(model, sample)
        return sample

    def fix_foot_penetration(self, sample, model, floor_pos_y=-1.0):
        right_kin_chain = [2, 5, 8]
        right_foot_ball = 3
        right_foot_toe = 2

        left_kin_chain = [1, 4, 7]
        left_foot_ball = 1
        left_foot_toe = 0


        right_toe_kin_chain = [2, 5, 8, 11]
        left_toe_kin_chain = [1, 4, 7, 10]

        sample = reloader.reload(model, sample)
        old_sample = sample.clone()
        fc_pred = sample[0, 150:154, 0, :]

        current_pos_dic = model.forward_kinematics(sample, None)
        current_pos = current_pos_dic["kp_45_joints"][:, :22].reshape(sample.shape[-1], 22,3) + current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_trans = current_pos_dic["pred_trans"].reshape(sample.shape[-1],3)

        print(current_pos[:, right_kin_chain[-1], :].min())
        print(current_pos[:, left_kin_chain[-1], :].min())

        # floor at -1.13
        # toe should be at -1.13
        # ankle should be at -1.13 + ( ankle - toe)
        print("setting floor y to ", floor_pos_y)
        print(current_pos[:, right_toe_kin_chain[-1], 1])
        for f in range(0, sample.shape[-1]):

            target = current_pos[f, right_kin_chain[-1], :]    
            if target[1] <= floor_pos_y:
                t = target.clone()
                t[1] = floor_pos_y
                sample[..., f] = self.forward_quat(sample, model, f, current_pos[f], current_trans[f], t, kin_chain=right_kin_chain).squeeze(-1)
            
            target = current_pos[f, left_kin_chain[-1], :]
            if target[1] <= floor_pos_y:
                t = target.clone()
                t[1] = floor_pos_y
                sample[..., f] = self.forward_quat(sample, model, f, current_pos[f], current_trans[f], t, kin_chain=left_kin_chain).squeeze(-1)


        sample = reloader.reload(model, sample)
        
        current_pos_dic = model.forward_kinematics(sample, None)
        current_pos = current_pos_dic["kp_45_joints"][:, :22].reshape(sample.shape[-1], 22,3) + current_pos_dic["pred_trans"].reshape(sample.shape[-1],3).unsqueeze(-2)
        current_trans = current_pos_dic["pred_trans"].reshape(sample.shape[-1],3)
        print(current_pos[:, right_toe_kin_chain[-1], 1])
        return sample


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
