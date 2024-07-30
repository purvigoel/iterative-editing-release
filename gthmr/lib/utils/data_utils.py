import torch
import mdm.utils.rotation_conversions as geometry
from VIBE.lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
import roma 
from misc_utils import to_np, to_tensor

def rotate_motion_by_rotmat(motion, rotmat):
    """
    A PyTorch function that rotates a global motion by rotmat.


    Input
        motion -- (N, 25, 6, T) motion where the last joint is XYZ000.
        rotmat -- (N, 3, 3) batch of rotation.
    """
    N, _, _, T = motion.shape

    # Root orientation.
    # Extract
    root_orient = motion[:, 0].permute(0, 2, 1)  # (N, T, 6)

    # Turn into aa
    root_orient_aa = geometry.rotation_6d_to_aa(root_orient)  # (N, T, 3)
    rot_aa = geometry.matrix_to_axis_angle(rotmat)  # (N, 3)

    # Apply rot
    new_root_orient_aa = roma.rotvec_composition(
        [rot_aa.unsqueeze(1).repeat(1, T, 1), root_orient_aa])

    # Turn back to 6d
    new_root_orient_rot6d = geometry.axis_angle_to_6d(
        new_root_orient_aa).permute(0, 2, 1)

    # Root translation.
    # Extract
    root_trans = motion[:, -1, :3].permute(0, 2,
                                           1).reshape(-1,
                                                      3)[...,
                                                         None]  # (N * T, 3, 1)

    # Apply rot
    new_root_trans = torch.bmm(
        rotmat.unsqueeze(1).repeat(1, T, 1, 1).reshape(-1, 3, 3),
        root_trans)[..., 0]  # (N * T, 3)
    new_root_trans = new_root_trans.reshape(N, T, 3)
    new_root_trans = torch.cat(
        [new_root_trans, torch.zeros_like(new_root_trans)],
        -1).permute(0, 2, 1)

    # create results
    new_motion = motion.clone()
    new_motion[:, 0] = new_root_orient_rot6d
    new_motion[:, -1] = new_root_trans

    return new_motion


def split_rot6d_extra(motion):
    """ 
    Get the rot6d motion (N,25,6,T) from data representations that have other 
    elements (rot6d_fc,rot6d_fc_shape). Assumed that the 25*6 rot6d values are 
    in the vector (N,25*6:,1,T). Also return `extra` which is the remaining 
    parts of the vector,  (N,:25*6,1,T)
    """
    motion = to_tensor(motion)
    N, J, D, T = motion.shape 
    # default rot6d format
    if (J,D)==(25,6):
        return motion, None
    # other valid data_reps: with some of: fc, shape, xyz position, velocity
    else:
        rot6d = motion[:,:150].permute(0,3,1,2).view(N,T,25,6).permute(0,2,3,1)
        extra = motion[:,150:]
        return rot6d, extra 

def combine_rot6d_extra(motion_6d, extra):
    """ 
    combine the parts that are created by `split_rot6d_extra` back into the
    formats rot6d_fc or rot6d_fc_shape """
    if extra is None: 
        return motion_6d
    motion_6d, extra = to_tensor(motion_6d), to_tensor(extra)
        
    N, J, D, T = motion_6d.shape 
    assert (J,D)==(25,6)
    
    motion_6d = motion_6d.permute(0,3,1,2).view(N,T,150,1).permute(0,2,3,1)
    motion_out = torch.cat((motion_6d, extra), dim=1)
    
    return motion_out

def get_xyz_from_motion(motion): 
    """ 
    """
    N,J,D,T = motion.shape
    assert J >= 236
    motion6d, extra = split_rot6d_extra(motion)
    xyz = extra[:,14:86] # (N,24*3,1,T)
    xyz = xyz.permute(0,3,1,2).reshape(N,T,24,3)
    xyz = xyz.permute(0,2,3,1)
    return xyz
