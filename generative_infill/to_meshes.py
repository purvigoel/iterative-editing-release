from jackson_import import *
from mdm.utils.fixseed import fixseed
import os
import argparse
import json
import numpy as np
import torch
from gthmr.emp_train.training_loop import TrainLoop
from mdm.utils.parser_util import train_ik_args, generate_args, train_args, train_emp_args
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
import pickle
import yaml
from utils.misc import updata_ns_by_missing_keys
from VIBE.lib.dataset.preprocess import Normalizer
from torch.autograd import Variable
import torch.optim as optim
from hmr.smplify.losses import angle_prior
from hmr.smplify.prior import create_prior
from VIBE.lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
import mdm.utils.rotation_conversions as geometry
#########################
from lib.utils.vis import render_image
import trimesh
import time

def fk(model, sample):
    N = sample.shape[0]
    T = sample.shape[-1]
    j_dic = model.forward_kinematics(sample, None)
    return j_dic, N, T

def dump_mesh(j_dic, save_file, batch, timestamp, N, T):
    transl = j_dic["pred_trans"].reshape(N, T,3)[batch, timestamp, :].cpu().numpy()

    smpl_joints = j_dic['kp_45_joints'][:, :24].reshape(N, T, 24, 3 ) +  j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)
    smpl_joints = smpl_joints +  torch.tensor([0, 0.2256, 0]).repeat(N, T, 1).unsqueeze(-2).to(smpl_joints.device)
    verts = j_dic['pred_vertices'][batch * T + timestamp].detach().cpu().numpy()
    verts = verts + transl[np.newaxis,...]
    faces = j_dic["faces"]
    #print("verts", verts.shape, "faces", faces.shape)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, maintain_order=True)
    mesh.export(save_file + "/" + str(timestamp).zfill(3) + ".obj") #, include_normals=True)

def to_meshes(model, body, out_folder):
    j_dic, N, T = fk(model, body)

    sample_id = 0
    os.makedirs(out_folder, exist_ok=True)
    start = time.time()
    #all_vn = []
    for i in range(T):
        vertex_norms = dump_mesh(j_dic, out_folder, sample_id, i, N, T)
    #    all_vn.append(vertex_norms)
    #all_vn = np.stack(all_vn)
    #print(all_vn.shape)
    #np.save(out_folder + "/fn.npy", all_vn)
    print("mesh write time", time.time() - start, T)


