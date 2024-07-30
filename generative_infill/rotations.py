from VIBE.lib.utils.geometry import rot6d_to_rotmat
from VIBE.lib.utils.geometry import rot6d_to_rotmat, batch_rodrigues
import mdm.utils.rotation_conversions as geometry
import math
import torch
import numpy as np

def transform_mat_waist(direc, ee, side, device, degrees):
    if direc == "flex":
        degrees *= -1
    aa = torch.tensor([1.0, 0.0, 0.0, 0.0 ,
            math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]).to(device)
    return aa


def transform_mat(direc,ee, side, device, degrees):
    if ee == "hand" and direc == "flex":
       if side == "right":
           degrees = degrees * -1
       degrees = degrees * -1
       aa = torch.tensor([
            math.cos(math.radians(degrees)), 0.0, math.sin(math.radians(degrees)), 0.0, 1.0, 0.0]).to(device)
       return aa
    elif ee == "hand" and direc == "extend":
       if side == "right":
           degrees = degrees * -1
       aa = torch.tensor([
            math.cos(math.radians(degrees)), 0.0, math.sin(math.radians(degrees)), 0.0, 1.0, 0.0]).to(device)
       return aa
    elif ee == "hand" and direc == "abduct":
       if side == "right":
           degrees = degrees * -1
       aa = torch.tensor([ 
            math.cos(math.radians(degrees)), -1 * math.sin(math.radians(degrees)), 0.0, math.sin(math.radians(degrees)), math.cos(math.radians(degrees)),0]).to(device)
       return aa
    elif ee == "hand" and direc == "rotate":
       if side == "left":
            degrees = degrees * -1
       aa = torch.tensor( [1.0, 0.0, 0.0, 0.0 ,
            math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]).to(device)
       return aa
    elif ee == "foot" and direc == "flex":
        #if side == "right":
        degrees = degrees * -1
        aa = torch.tensor([1.0, 0.0, 0.0, 0.0 , 
            math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]).to(device)
        return aa
    elif ee == "foot" and  direc == "extend":
        #if side == "right":
        degrees = degrees * -1
        aa = torch.tensor([1.0, 0.0, 0.0, 0.0 , 
            math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]).to(device)
        return aa
    elif ee == "foot" and direc == "abduct":
       if side == "right":
           degrees = degrees * -1
       aa = torch.tensor([
            math.cos(math.radians(degrees)), -1 * math.sin(math.radians(degrees)), 0.0, math.sin(math.radians(degrees)), math.cos(math.radians(degrees)),0]).to(device)
       return aa
    elif ee == "hips":
       aa = torch.tensor([1.0, 0.0, 0.0, 0.0 ,
            math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]).to(device)
       return aa
    elif ee == "back":
       aa = torch.tensor([1.0, 0.0, 0.0, 0.0 ,
            math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]).to(device)

       return aa


def transform(input_motion, transform, joints, keyframe):
    rotation_rot6d = input_motion[joints, 0].permute(1,0)
    aa_matrix = geometry.rotation_6d_to_aa(rotation_rot6d)
    matrix = batch_rodrigues(aa_matrix).view(-1, 3, 3)
    matrix_transform = torch.bmm(matrix , transform.repeat(input_motion.shape[-1], 1, 1))  
    matrix_transform = matrix_transform.permute(1, 2, 0).unsqueeze(0)

    keyframe[:, joints[0], :, :] = matrix_transform[:, 0, 0, :]
    keyframe[:, joints[1], :, :] = matrix_transform[:, 0, 1, :]
    keyframe[:, joints[2], :, :] = matrix_transform[:, 0, 2, :]
    keyframe[:, joints[3], :, :] = matrix_transform[:, 1, 0, :]
    keyframe[:, joints[4], :, :] = matrix_transform[:, 1, 1, :]
    keyframe[:, joints[5], :, :] = matrix_transform[:, 1, 2, :]
    return keyframe
