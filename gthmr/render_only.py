from jackson_import import *
from mdm.utils.fixseed import fixseed
import os
import argparse
import json
import numpy as np
import torch
from gthmr.emp_train.training_loop import TrainLoop
from mdm.utils.parser_util import generate_args, train_args, train_emp_args
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from gthmr.emp_train.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
from mdm.utils.model_util import create_emp_model_and_diffusion
import shutil
from mdm.data_loaders.tensors import collate
from gthmr.lib.utils.mdm_utils import viz_motions
from VIBE.lib.dataset.vibe_dataset import rotate_about_D
from gthmr.emp_train.get_data import get_dataset_loader_dict_new2
from gthmr.lib.utils import data_utils
import ipdb
import yaml
from utils.misc import updata_ns_by_missing_keys
from mdm.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_aa, axis_angle_to_6d
import generative_infill.asset_library as asset_library
import generative_infill.dataset_gen_summary_statistics as dataset_gen
from generative_infill.reloader import reload
from generative_infill.to_meshes import to_meshes
import generative_infill.ik_solver2 as IKSolver
#########################
# Load pretrained MR-DM #
#########################

args = train_emp_args()
fixseed(args.seed )

device = 'cuda'

os.makedirs(args.save_dir, exist_ok=True)


def interpolate(to_infill, keyframe, frame, window, hi_window=None):
    og_motion = to_infill.clone()
    frames = [frame]
    for frame in frames:
        keyf = keyframe[:, :, :, frame]

        delta = keyf.clone() # - og_motion[:,:,:,frame].clone()
        lo = max(0, frame - window)
        if not hi_window:
            hi_window = window
        hi = min(to_infill.shape[-1] - 1, frame + hi_window)
        print(lo, hi)
        for fr in range(lo, frame):
            a = (fr - lo) / (frame - lo)
            og_motion[:, :, :, fr] = og_motion[:, :, :, lo] * (1-a)  + (a) * (delta)

        if frame != 59:
            for fr in range(frame, hi):
                a = (fr - frame) / (hi - frame)
                og_motion[:, :, :, fr] = og_motion[:, :, :, hi] * a  + (1 - a) * (delta)
        else:
            og_motion[:, :, :, frame] = delta
    to_infill[:, :25*6, : , :] = og_motion[:, :25*6, : , :].clone()
    to_infill[:, 164:, :, :] = og_motion[:, 164:, :, :].clone()
    return to_infill

def interpolatemask(lo, hi, mid):
    mask = torch.ones( 1, 60)
    frame = mid
    for fr in range(lo, frame):
         a = (fr - frame) / (hi - frame)
         mask[..., fr] = a
    for fr in range(frame, hi):
        a = (fr - frame) / (hi - frame)
        mask[..., fr] = 1-a
    return mask

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

def reload2(input_motions, num_motions=1, sample_id=0, include_fc=True):
        T = input_motions.shape[-1]
        N = input_motions.shape[0]
        #j_dic = self.fk_model.forward_kinematics(input_motions, None)
        #j_dic['kp_45_joints'] = j_dic['kp_45_joints'].to(self.device)
        #smpl_joints = j_dic['kp_45_joints'][:, :24].reshape(N, T, 24, 3 ) +  j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)
        #smpl_joints = smpl_joints +  torch.tensor([0, 0.2256, 0]).repeat(N, T, 1).unsqueeze(-2).to(smpl_joints.device)

        positions = input_motions[:, 164:, :, :].clone()
        positions = positions.permute(0, 3, 1, 2).squeeze(-1).reshape(N, T, 24, 3)
        smpl_joints = positions
        if include_fc and T > 1:
            for_vel = smpl_joints.clone()
            target_vel, target_vel_xyz = comptue_velocity(for_vel[sample_id])
            _, fc = estimate_foot_contact(target_vel, 0.03)
            fc = fc.float()
            input_motions[:, 150:154, :, :] = fc.permute( 1, 0).unsqueeze(0).unsqueeze(2).repeat(N, 1, 1, 1)

        smpl_joints = smpl_joints.view(N, T, 24 * 3, 1)
        smpl_joints = smpl_joints.permute(0, 2, 3, 1)
        #input_motions[:, 164 : 236, :, :] = smpl_joints.clone()
        return input_motions

def interpolate_loop(to_infill, frames):
    og_motion = to_infill.clone()
    for fr_ind in range(len(frames)):
        if fr_ind == 0:
            prev_frame = 0
        else:
            prev_frame = frames[fr_ind - 1]

        if fr_ind == len(frames) - 1:
            next_frame = 59
        else:
            next_frame = frames[fr_ind + 1]

        frame = frames[fr_ind]

        keyf = to_infill[:, :, :, frame]
        delta = keyf.clone()
        lo = prev_frame #frame - prev_frame
        hi =  next_frame# - frame

        for fr in range(lo, frame):
            a = (fr - lo) / (frame - lo)
            og_motion[:, :, :, fr] = og_motion[:, :, :, lo] * (1-a)  + (a) * (delta)

        for fr in range(frame, hi):
            a = (fr - frame) / (hi - frame)
            og_motion[:, :, :, fr] = og_motion[:, :, :, hi] * a  + (1 - a) * (delta)
    to_infill[:, :25*6, : , :] = og_motion[:, :25*6, : , :].clone()
    to_infill[:, 164:, :, :] = og_motion[:, 164:, :, :].clone()
    return to_infill

# # Load and parse data config
# data_cfg = yaml.safe_load(open(args.data_config_path, 'r'))

# _, eval_data_loaders_dic, args.total_batch_size = get_dataset_loader_dict_new2(
#         data_cfg, eval_only=True)

path_model_args = os.path.join(os.path.dirname(args.model_path), "args.json")
if not os.path.exists(path_model_args):
    raise ValueError(f"Model path [{args.model_path}] must be in the same" \
                    "directory as its model args file: [args.json]")
with open(path_model_args, 'r') as f:
    args_pretrained_model = argparse.Namespace(**json.load(f))

args_pretrained_model.total_batch_size = 5

# Overwrite save_dir
args_pretrained_model.save_dir = args.save_dir

args_pretrained_model.dataset = "amass_hml_keyframe"

# Backward comp
args_pretrained_model = updata_ns_by_missing_keys(args_pretrained_model, args)

# Load Pretrained Model
print("creating model and diffusion...")
model, diffusion = create_emp_model_and_diffusion(args_pretrained_model, None)
model.to(device)
model.rot2xyz.smpl_model.eval()

print(f"Loading checkpoints from [{args.model_path}]...")
state_dict = torch.load(args.model_path)
load_model_wo_clip(model, state_dict)
num_samples = 50
max_frames = 60
args.batch_size = num_samples


##############################
# Visualize generated motion #
##############################
N = num_samples  # batch size
T = max_frames  # this is fixed by the model
video_features = torch.zeros(N, T, 2048)

#FILENAME = "ASE/ase/data/smpl_motions/attack_combo_smpl_params.npy_amass.npy" #"final_results_siggraph_may6/editing/second_kick/sampletest.npy"
FILENAME = "sampletest.npy"
ID = 18
ID = ID // 2 
sample = torch.tensor(np.load(FILENAME))[..., :60]

def gauss_smooth(sample, sigma=1):
    import scipy.ndimage
    sample = scipy.ndimage.gaussian_filter1d(sample, sigma, axis=-1)
    return torch.tensor(sample)


iksolver = IKSolver.IKSolver()
sample = sample[ID].unsqueeze(0)
#sample = gauss_smooth(sample.cpu(), 0.25)
#sample = iksolver.fix_foot_penetration(sample, model, -1.0)
#sample = iksolver.fix_foot_slide(sample, model, 0, "left", force=False)
#sample = iksolver.fix_foot_slide(sample, model, 0, "right", force=False)

N = 1
j_dic = model.forward_kinematics(sample, None)
smpl_joints = j_dic['kp_45_joints'][:, :22].reshape(N, T, 22, 3) + j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)

np.save("smpl_joints.npy", smpl_joints.cpu().numpy())

ID = 0
to_meshes(model, sample[ID].unsqueeze(0).cpu(), "/raid/pgoel2/bio-pose/user_study_siggraph/out/")
