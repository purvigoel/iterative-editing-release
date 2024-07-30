import torch
import numpy as np

def comptue_velocity(target_xyz):
    # estimate vel naively
    target_vel_xyz = target_xyz[1:] - target_xyz[:-1]  # (N-1,24,3)
    # copy the last value to get equal lengths
    target_vel_xyz = torch.cat((target_vel_xyz, target_vel_xyz[[-1]]),
                                       0)  # (N,24,3)
    # norm of each velocity vector
    target_vel = torch.linalg.norm(target_vel_xyz, axis=-1)  # (N,24)

    return target_vel, target_vel_xyz

def estimate_foot_contact(target_vel, foot_vel_threshold):
    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11

    relevant_joints = [
                l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx
    ]
    target_vel = target_vel[:, relevant_joints]
    fc_mask = (target_vel <= foot_vel_threshold)

    return target_vel, fc_mask

def reload(model, input_motions, num_motions=1, sample_id=0, include_fc=True):
    T = input_motions.shape[-1]
    N = input_motions.shape[0]
    j_dic = model.forward_kinematics(input_motions, None)

    smpl_joints = j_dic['kp_45_joints'][:, :24].reshape(N, T, 24, 3 ) +  j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)
    smpl_joints = smpl_joints +  torch.tensor([0, 0.2256, 0]).repeat(N, T, 1).unsqueeze(-2).to(smpl_joints.device)

    if include_fc and T > 1:
        for_vel = smpl_joints.clone()
        target_vel, target_vel_xyz = comptue_velocity(for_vel[sample_id])
        _, fc = estimate_foot_contact(target_vel, 0.03)
        fc = fc.float()
        input_motions[:, 150:154, :, :] = fc.permute( 1, 0).unsqueeze(0).unsqueeze(2).repeat(N, 1, 1, 1)

    smpl_joints = smpl_joints.view(N, T, 24 * 3, 1)


    smpl_joints = smpl_joints.permute(0, 2, 3, 1)
    input_motions[:, 164 : 236, :, :] = smpl_joints.clone()
    return input_motions
