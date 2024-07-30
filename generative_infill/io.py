import numpy as np
from meo_constants import meoConstant
import torch

def save_files(model, dir, sample, condition, name, num_motions=1):
    N = num_motions
    T = sample.shape[-1]
    #if T == 1:
    #   sample = sample.repeat(1,1,1,meoConstant.max_frames)
    #  T = meoConstant.max_frames
    sample = torch.tensor(sample)
    condition = torch.tensor(condition)

    np.save(dir + "/" + name + ".npy", sample.cpu().numpy())
    np.save(dir + "/" + name + "_condition.npy", condition.cpu().numpy())
    j_dic = model.forward_kinematics(sample, None)
    smpl_joints = j_dic['kp_45_joints'][:, :22].reshape(N, T, 22, 3)  +  j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)
    
    first_iter_joints = smpl_joints
    print(dir + "/" + name + "_iter_joints.npy")
    np.save(dir + "/" + name + "_iter_joints.npy", first_iter_joints.cpu().numpy())

    T = condition.shape[-1]
    #if T == 1:
    #    condition = condition.repeat(1,1,1,meoConstant.max_frames)
    #    T = meoConstant.max_frames

    condition_dic = model.forward_kinematics(condition, None)
    condition_joints = condition_dic['kp_45_joints'][:, :22].reshape(N, T, 22, 3) + condition_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)
    np.save(dir + "/" + name + "_iter_condition_joints.npy", condition_joints.cpu().numpy())

    all_joints = []
    for i in range(N):
        all_joints.append(condition_joints.cpu().numpy()[i])
        all_joints.append(first_iter_joints.cpu().numpy()[i])
    
    all_joints = np.stack(all_joints, axis=0)
    np.save(dir + "/" + name + "_all.npy", all_joints)
    
