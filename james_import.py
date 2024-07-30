import os
import json
import ipdb
import torch
from mdm.utils.fixseed import fixseed
from mdm.utils.parser_util import train_emp_args
from mdm.utils import dist_util
from gthmr.emp_train.training_loop import TrainLoop
from gthmr.emp_train.get_data import get_dataset_loader_dict
from mdm.utils.model_util import create_emp_model_and_diffusion, load_model_wo_clip
from mdm.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from nemo.utils.exp_utils import create_latest_child_dir
import yaml
from gthmr.lib.utils.mdm_utils import viz_motions
from gthmr.lib.models.emp import rotate_motion_by_rotmat
from gthmr.lib.models import emp
from mdm.model.rotation2xyz import Rotation2xyz
from nemo.utils.misc_utils import to_np, to_tensor
from VIBE.lib.core.config import VIBE_DATA_DIR
from VIBE.lib.data_utils.kp_utils import convert_kps
from VIBE.lib.dataset.vibe_dataset import rotate_points_3d
from VIBE.lib.dataset import vibe_dataset
from VIBE.lib.utils.eval_utils import (
    compute_accel, compute_error_accel, compute_error_verts,
    batch_compute_similarity_transform_torch, compute_mpjpe, compute_g_mpjpe)


import ipdb; ipdb.set_trace()
    # out_path = "/pasteur/u/jmhb/bio-pose/gthmr/results/tmp"
import os
from mdm.model.rotation2xyz import Rotation2xyz
rot2xyz = Rotation2xyz(device="cuda", dataset=None)
device='cuda'
def to_xyz(motions):
    with torch.no_grad():
        motions_xyz =  rot2xyz(x=motions.to(device), mask=None, pose_rep='rot6d', glob=True, translation=True,
               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
            get_rotations_back=False, get_fc_back=False).cpu()
    return motions_xyz

1motion_6d_rot_xyz = rot2xyz(motion_6d_rot) 