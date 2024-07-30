# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import argparse
from yacs.config import CfgNode as CN
import os
import os.path as osp

ROOT = osp.join(os.environ['BIO_POSE_ROOT'], 'VIBE')
# CONSTANTS
# You may modify them at will
VIBE_DB_DIR = osp.join(ROOT, 'data/vibe_db')
AMASS_DIR = osp.join(ROOT, 'data/amass')
INSTA_DIR = osp.join(ROOT, 'data/insta_variety')
MPII3D_DIR = osp.join(ROOT, 'data/mpi_inf_3dhp')
THREEDPW_DIR = osp.join(ROOT, 'data/3dpw')
PENNACTION_DIR = osp.join(ROOT, 'data/penn_action')
POSETRACK_DIR = osp.join(ROOT, 'data/posetrack')
VIBE_DATA_DIR = osp.join(ROOT, 'data/vibe_data')

# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = True
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 8
cfg.DEBUG_FREQ = 1000
cfg.SEED_VALUE = -1

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.DATASETS_2D = ['Insta']
cfg.TRAIN.DATASETS_3D = ['MPII3D']
cfg.TRAIN.DATASET_EVAL = 'ThreeDPW'
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.DATA_2D_RATIO = 0.5
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 5
cfg.TRAIN.PRETRAINED_REGRESSOR = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.RESUME = ''
cfg.TRAIN.NUM_ITERS_PER_EPOCH = 1000
cfg.TRAIN.LR_PATIENCE = 5

# <====== generator optimizer
cfg.TRAIN.GEN_OPTIM = 'Adam'
cfg.TRAIN.GEN_LR = 1e-4
cfg.TRAIN.GEN_WD = 1e-4
cfg.TRAIN.GEN_MOMENTUM = 0.9

# <====== motion discriminator optimizer
cfg.TRAIN.MOT_DISCR = CN()
cfg.TRAIN.MOT_DISCR.OPTIM = 'SGD'
cfg.TRAIN.MOT_DISCR.LR = 1e-2
cfg.TRAIN.MOT_DISCR.WD = 1e-4
cfg.TRAIN.MOT_DISCR.MOMENTUM = 0.9
cfg.TRAIN.MOT_DISCR.UPDATE_STEPS = 1
cfg.TRAIN.MOT_DISCR.FEATURE_POOL = 'concat'
cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE = 1024
cfg.TRAIN.MOT_DISCR.NUM_LAYERS = 1
cfg.TRAIN.MOT_DISCR.ATT = CN()
cfg.TRAIN.MOT_DISCR.ATT.SIZE = 1024
cfg.TRAIN.MOT_DISCR.ATT.LAYERS = 1
cfg.TRAIN.MOT_DISCR.ATT.DROPOUT = 0.1

cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 20
cfg.DATASET.OVERLAP = 0.5

cfg.LOSS = CN()
cfg.LOSS.KP_2D_W = 60.
cfg.LOSS.KP_3D_W = 30.
cfg.LOSS.KP_3D_VEL_W = 0.
cfg.LOSS.SHAPE_W = 0.001
cfg.LOSS.POSE_W = 60.0
cfg.LOSS.D_MOTION_LOSS_W = 1.
cfg.LOSS.N_JOINTS = 49

cfg.MODEL = CN()

cfg.MODEL.TEMPORAL_TYPE = 'gru'
cfg.MODEL.DONT_USE_DECODER = False

# GRU model hyperparams
cfg.MODEL.TGRU = CN()
cfg.MODEL.TGRU.NUM_LAYERS = 1
cfg.MODEL.TGRU.ADD_LINEAR = False
cfg.MODEL.TGRU.RESIDUAL = False
cfg.MODEL.TGRU.HIDDEN_SIZE = 2048
cfg.MODEL.TGRU.BIDIRECTIONAL = False
cfg.MODEL.PREDICTS_ROOT_TRAJ = False
cfg.MODEL.REGRESSOR_CLASS = 'MyRegressor'
cfg.MODEL.PREDICT_RIGID = False
cfg.MODEL.WEIGHT_EXTRA_LOSS = 0.
cfg.MODEL.DECODER_PATH = 'mdm/save/unconstrained/model000450000.pt'
cfg.MODEL.INIT_POSE_GAIN = 0.01
cfg.MODEL.NORMALIZE_TIME_ONLY = True
cfg.MODEL.USE_DPM_SOLVER = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    # Arguments to overwrite
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--num_iters_per_epoch', type=int, default=100)
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--dont_use_decoder', type=int, default=-1)
    parser.add_argument('--weight_extra_loss', type=float, default=0)
    parser.add_argument('--init_pose_gain', type=float, default=0.01)
    parser.add_argument('--weight_3d_kp_vel', type=float, default=0.0)
    parser.add_argument('--normalize_time_only', type=int, default=1)
    parser.add_argument('--use_dpm_solver', type=int, default=1)
    parser.add_argument('--loss_pose_w', type=float, default=-1)
    parser.add_argument('--loss_n_joints', type=int, default=-1)

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.pretrained is not None:
        cfg.TRAIN.PRETRAINED = args.pretrained

    if args.lr != -1:
        cfg.TRAIN.GEN_LR = args.lr

    if args.dont_use_decoder != -1:
        cfg.MODEL.DONT_USE_DECODER = True

    if args.weight_extra_loss > 0:
        cfg.MODEL.WEIGHT_EXTRA_LOSS = args.weight_extra_loss

    if args.init_pose_gain != 0.01:
        cfg.MODEL.INIT_POSE_GAIN = args.init_pose_gain

    if args.num_iters_per_epoch != 100:
        cfg.TRAIN.NUM_ITERS_PER_EPOCH = args.num_iters_per_epoch

    if args.weight_3d_kp_vel > 0.:
        cfg.LOSS.KP_3D_VEL_W = args.weight_3d_kp_vel

    cfg.MODEL.NORMALIZE_TIME_ONLY = args.normalize_time_only
    cfg.MODEL.USE_DPM_SOLVER = args.use_dpm_solver

    if args.loss_pose_w != -1:
        cfg.LOSS.POSE_W = args.loss_pose_w

    if args.loss_n_joints != -1:
        cfg.LOSS.N_JOINTS = args.loss_n_joints

    return args, cfg, cfg_file
