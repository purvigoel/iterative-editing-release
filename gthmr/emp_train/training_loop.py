# start regular training from a pretrained model
import copy
import functools
import os
import os.path as osp
import time
from types import SimpleNamespace
import numpy as np
import ipdb
import blobfile as bf
import torch
from torch.optim import AdamW

from mdm.diffusion import logger
from mdm.utils import dist_util
from mdm.diffusion.fp16_util import MixedPrecisionTrainer
from mdm.diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from mdm.diffusion.resample import create_named_schedule_sampler
from mdm.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from mdm.eval import eval_humanml, eval_humanact12_uestc
from mdm.data_loaders.get_data import get_dataset_loader
from VIBE.lib.utils.utils import move_dict_to_device
from misc_utils import to_np, to_tensor
from VIBE.lib.data_utils.kp_utils import convert_kps
from VIBE.lib.utils.eval_utils import (
    compute_accel, compute_error_accel, compute_error_verts,
    batch_compute_similarity_transform_torch, compute_mpjpe, compute_g_mpjpe,
    compute_error_g_vel, compute_error_g_acc, compute_nemo_mpjpe,
    compute_error_g_verts, f_target_vert)
import pandas as pd
import roma
from gthmr.lib.utils.data_utils import split_rot6d_extra, combine_rot6d_extra, rotate_motion_by_rotmat

# HMR related
from gthmr.lib.core.loss import VIBELoss
#
import mdm.utils.rotation_conversions as geometry
from collections import defaultdict


def log_dict(dic, fpath):
    print("Logging to: ", fpath)
    if not osp.exists(fpath):
        new_dic = dict([(k, [v]) for k, v in dic.items()])
        df = pd.DataFrame(new_dic)
        df.to_csv(fpath)
    else:
        old_df = pd.read_csv(fpath, index_col=0)
        new_dic = dict([(k, [v]) for k, v in dic.items()])
        df = pd.DataFrame(new_dic)
        old_df = old_df.append(df, 1)
        old_df.to_csv(fpath)


def remove_motion_rotation(motion, data_name='h36m', data_rep='rot6d'):
    """ 
    Util for undoing the change in orientation done to the 25d data reps, or the 
    data reps that use contain it. 
    """
    assert motion.ndim == 4
    N, J, D, T = motion.shape
    if data_name in ('h36m', '3dpw', 'nemomocap', 'nemomocap2'):
        motion_6d, motion_extra = split_rot6d_extra(motion)

        # rotate the 6d motion component
        rotmat = roma.rotvec_to_rotmat(torch.tensor([-np.pi, 0, 0]))
        rotmat = rotmat.unsqueeze(0).repeat(N, 1, 1).to(motion_6d.device)
        motion_6d_rot = rotate_motion_by_rotmat(motion_6d, rotmat)

        # reattach the non-5d components
        if data_rep == 'rot6d':
            motion_out = motion_6d_rot
        else:
            motion_out = combine_rot6d_extra(motion_6d_rot, motion_extra)

    else:
        motion_out = motion

    return motion_out


class TrainLoop:

    def __init__(self,
                 args,
                 train_platform,
                 model,
                 diffusion,
                 train_data_loaders_dic,
                 eval_data_loaders_dic,
                 eval_only=False):
        self.args = args
        self.eval_only = eval_only
        assert args.dataset == ''
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        if model is not None:
            self.cond_mode = model.cond_mode
        # self.data = data
        self.train_hmr_data_loader_dict = None
        self.train_motion_data_loader_dict = None
        self.eval_hmr_data_loader_dict = None
        self.train_hmr_data_iter_dict = None
        self.train_motion_data_iter_dict = None

        if train_data_loaders_dic is not None:
            if train_data_loaders_dic['hmr'] is not None:
                self.train_hmr_data_loader_dict = {}
                self.train_hmr_data_iter_dict = {}
                for name in train_data_loaders_dic['hmr']:
                    self.train_hmr_data_loader_dict[
                        name] = train_data_loaders_dic['hmr'][name]
                    self.train_hmr_data_iter_dict[name] = iter(
                        self.train_hmr_data_loader_dict[name])
            if train_data_loaders_dic['motion'] is not None:
                self.train_motion_data_loader_dict = {}
                self.train_motion_data_iter_dict = {}
                for name in train_data_loaders_dic['motion']:
                    self.train_motion_data_loader_dict[
                        name] = train_data_loaders_dic['motion'][name]
                    self.train_motion_data_iter_dict[name] = iter(
                        self.train_motion_data_loader_dict[name])

        if eval_data_loaders_dic['hmr'] is not None:
            self.eval_hmr_data_loader_dict = eval_data_loaders_dic['hmr']

        self.train_data_loaders_dic = train_data_loaders_dic
        self.batch_size = args.total_batch_size
        self.microbatch = args.total_batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = args.num_steps

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        if not self.eval_only:
            if train_data_loaders_dic is not None:
                # if self.train_hmr_data_loader is not None:
                #     len_loader = len(self.train_hmr_data_loader)
                # elif self.train_motion_data_loader is not None:
                #     len_loader = len(self.train_motion_data_loader)
                # else:
                #     raise
                self.num_iters_per_epoch = args.num_iters_per_epoch
                self.num_epochs = self.num_steps // self.num_iters_per_epoch + 1
                # self.num_iters_per_epoch = len_loader

            self.sync_cuda = torch.cuda.is_available()

            self._load_and_sync_parameters()
            self.mp_trainer = MixedPrecisionTrainer(
                model=self.model,
                use_fp16=self.use_fp16,
                fp16_scale_growth=self.fp16_scale_growth,
            )

            self.opt = AdamW(self.mp_trainer.master_params,
                             lr=self.lr,
                             weight_decay=self.weight_decay)

            self.hmr_params = list(self.model.pitch_corrector.parameters())
            self.hmr_opt = AdamW(self.hmr_params,
                                 lr=self.lr,
                                 weight_decay=self.weight_decay)
            self.hmr_loss = VIBELoss(
                e_loss_weight=0,  # 2D kp loss
                e_3d_loss_weight=self.args.e_3d_loss_weight,
                e_3d_vel_loss_weight=0,  # not used.
                e_pose_loss_weight=self.args.e_pose_loss_weight,
                e_shape_loss_weight=self.args.e_shape_loss_weight,
                n_joints=49)

            if self.resume_step:
                self._load_optimizer_state()
                # Model was resumed, either due to a restart or a checkpoint
                # being specified at the command line.

            self.schedule_sampler_type = 'uniform'
            self.schedule_sampler = create_named_schedule_sampler(
                self.schedule_sampler_type, diffusion)

        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset,
                                            batch_size=args.eval_batch_size,
                                            num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(
                name=args.dataset,
                batch_size=args.eval_batch_size,
                num_frames=None,
                split=args.eval_split,
                hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset,
                                                    dist_util.dev())
            self.eval_data = {
                'test':
                lambda: eval_humanml.get_mdm_loader(
                    model,
                    diffusion,
                    args.eval_batch_size,
                    gen_loader,
                    mm_num_samples,
                    mm_num_repeats,
                    gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples,
                    scale=1.,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model

        # HMR
        self.evaluation_accumulators = dict.fromkeys([
            'pred_j3d', 'pred_j3d_nobeta', 'target_j3d', 'target_j3d_mosh',
            'target_j3d_mosh_nobeta', 'target_theta', 'pred_verts',
            'pred_slv_j3d', 'target_slv_j3d', 'pred_g_j3d', 'target_g_j3d',
            'pred_g_j3d_nobeta', 'target_g_j3d_mosh',
            'target_g_j3d_mosh_nobeta', 'vid_name', 'idxs', 'motions',
            'pred_motions', 'img_name', 'pred_g_v3d', 'pred_g_v3d_nobeta',
            'target_g_v3d', 'target_theta', 'orig_trans'
        ])

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(
                resume_checkpoint)
            logger.log(
                f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(resume_checkpoint,
                                          map_location=dist_util.dev()))

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint),
                                 f"opt{self.resume_step:09}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def _use_img_name_to_index_data(self, img_name, all_data, dataset_name):
        """
        A helper function for `validate_baseline`.  

        Input
            img_name -- (N, T, 1) img_names
            all_data -- a dictionary of baseline prediction
        Output
            pred_motion -- None, or (N, 25, 6, T) the standard format
        """
        N, T, _ = img_name.shape
        # assert N == 1
        B = N * T
        print(N, T)
        all_data_img_name = list(all_data['img_name'][:, 0])
        # Get the corresponding indices
        idx_arr = np.zeros((N * T, )).astype('int32')
        for n, cur_img_name in enumerate(img_name.ravel()):
            if dataset_name in ('h36m'):
                cur_img_name = osp.basename(cur_img_name)

            try:
                idx_arr[n] = all_data_img_name.index(cur_img_name)
            except ValueError:
                print('Missing: ', cur_img_name)
                return None

        prefix = all_data['prefix']
        pose = all_data[f'{prefix}_pose_batch'][idx_arr]
        orient = all_data[f'{prefix}_orient_batch'][idx_arr]
        theta = to_tensor(
            np.hstack([orient, pose]).reshape(B, 24,
                                              3).reshape(-1, 3))  # (B, 24, 3)
        theta6d = geometry.axis_angle_to_6d(theta).reshape(B, 24, 6).reshape(
            N, T, 24, 6)
        trans = to_tensor(all_data[f'{prefix}_trans_batch'][idx_arr]).reshape(
            N, T, 3)
        motion = torch.zeros((N, 25, 6, T)).cuda()
        motion[:, :24] = theta6d.permute(0, 2, 3, 1)
        motion[:, 24, :3] = trans.permute(0, 2, 1)
        beta = to_tensor(all_data[f'{prefix}_beta'][idx_arr]).reshape(N, T, 10)
        return motion, beta

    def validate_baseline(self, all_data, n_batches=None):
        """
        Analogous to `validate_hmr`, but for baseline prediction `all_data`. 

        n_batches: restrict the number of batches to loop throgh. Used for debugging.
        """
        if n_batches == -1:
            n_batches = None
        # if self.eval_hmr_data_loader is None:
        #     print("Skipping eval ... no dataloader found")
        #     return
        assert len(self.eval_hmr_data_loader_dict) == 1

        print('  >> Accumulating predictionf for HMR evaluation.')

        # Refresh accumulator
        if self.evaluation_accumulators is not None:
            for k, v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        loader = list(self.eval_hmr_data_loader_dict.values())[0]
        data_name = loader.dataset.dataset
        data_rep = loader.dataset.data_rep

        miss_count = 0
        for i, eval_hmr_batch in tqdm(enumerate(loader), total=len(loader)):
            idxs = np.arange(len(loader)) + i * len(loader)
            if n_batches is not None and i >= n_batches:
                break

            hmr_motion, hmr_cond = eval_hmr_batch
            hmr_motion = hmr_motion.cuda()
            n, j, d, t = hmr_motion.shape

            img_name = hmr_cond['y']['img_name']
            pred = self._use_img_name_to_index_data(img_name, all_data,
                                                    loader.dataset.dataset)
            if pred is None:
                miss_count += 1
                print("N misses: ", miss_count)
                continue
            else:
                pred_motion, pred_beta = pred
                pred_beta = pred_beta[:,
                                      0]  # Take only the beta from 1st frame.
                pred_motion = to_tensor(pred_motion)
                pred_beta = to_tensor(pred_beta)
            with torch.no_grad():
                pred_motion_rot = pred_motion.clone()
                # prediction kps without beta/shape
                pred_j3d_nobeta_dict = self.model.forward_kinematics(
                    pred_motion_rot, None)
                pred_j3d_nobeta = to_np(pred_j3d_nobeta_dict['kp_c14_joints']
                                        )  # relative keypoints
                pred_j3d_trans_nobeta = to_np(
                    pred_j3d_nobeta_dict['pred_trans'])  # translation
                pred_g_j3d_nobeta = pred_j3d_nobeta + pred_j3d_trans_nobeta  # global keypoints
                pred_g_j3d_nobeta = pred_g_j3d_nobeta.reshape(
                    n, t, 14, 3)  # reshape to include time for compute_g_mpjpe
                # prediction kps with beta/shape

                pred_j3d_beta_dict = self.model.forward_kinematics(
                    pred_motion_rot, pred_beta)
                pred_j3d_beta = to_np(
                    pred_j3d_beta_dict['kp_c14_joints'])  # relative keypoints
                pred_j3d_trans_beta = to_np(
                    pred_j3d_beta_dict['pred_trans'])  # translation
                pred_g_j3d_beta = pred_j3d_beta + pred_j3d_trans_beta  # global keypoints
                pred_g_j3d_beta = pred_g_j3d_beta.reshape(
                    n, t, 14, 3)  # needed for compute_g_mpjpe

                ## global vertices for rendering
                # global vertices wo beta
                pred_v3d_nobeta = to_np(pred_j3d_nobeta_dict['pred_vertices'])
                g_pred_v3d_nobeta = pred_v3d_nobeta + to_np(
                    pred_j3d_nobeta_dict['pred_trans'])
                g_pred_v3d_nobeta = g_pred_v3d_nobeta.reshape(n, t, -1, 3)
                # global vertices with beta
                pred_v3d_beta = to_np(pred_j3d_beta_dict['pred_vertices'])
                g_pred_v3d_beta = pred_v3d_nobeta + to_np(
                    pred_j3d_beta_dict['pred_trans'])
                g_pred_v3d_beta = g_pred_v3d_beta.reshape(n, t, -1, 3)

            ## mocap target keypoints
            if 'gt_spin_joints3d' in hmr_cond['y']:
                target_j3d_mocap = to_np(
                    hmr_cond['y']['gt_spin_joints3d'])[..., :3]  # (N,T,49,3)
                n, t, j, _ = target_j3d_mocap.shape
                target_j3d_mocap = convert_kps(
                    target_j3d_mocap.reshape(n * t, j, 3), 'spin', 'common')
                target_g_j3d_mocap = target_j3d_mocap.reshape(
                    n, t, 14,
                    3)  # global kps are the same data, but reshaped (n,t,14,3)

            ## mosh target keypoints: both with and without gt beta
            with torch.no_grad():
                # gt motion was previously rotated - remove it
                hmr_motion_rot = remove_motion_rotation(hmr_motion.clone(),
                                                        data_name=data_name,
                                                        data_rep=data_rep)
                # kps without beta: local and global
                target_j3d_mosh_nobeta_dict = self.model.forward_kinematics(
                    hmr_motion_rot, None)
                target_j3d_mosh_nobeta = to_np(
                    target_j3d_mosh_nobeta_dict['kp_c14_joints'])
                target_j3d_mosh_nobeta += to_np(
                    target_j3d_mosh_nobeta_dict['pred_trans'])
                target_g_j3d_mosh_nobeta = target_j3d_mosh_nobeta.reshape(
                    n, t, 14,
                    3)  # global kps are the same data, but reshaped (n,t,14,3)
                # kps without beta: local and global
                target_beta = hmr_cond['y']['theta'][..., -10:].to(
                    self.device).float()  # (N,T,10)
                target_j3d_mosh_beta_dict = self.model.forward_kinematics(
                    hmr_motion_rot, target_beta)
                target_j3d_mosh_beta = to_np(
                    target_j3d_mosh_beta_dict['kp_c14_joints'])
                target_j3d_mosh_beta += to_np(
                    target_j3d_mosh_beta_dict['pred_trans'])
                target_g_j3d_mosh_beta = target_j3d_mosh_beta.reshape(
                    n, t, 14,
                    3)  # global kps are the same data, but reshaped (n,t,14,3)

            ## for common14 joints, h36m doesn't have the last one, so remove them
            if loader.dataset.dataset == 'h36m':
                # mocap targets: local and global
                target_j3d_mocap = target_j3d_mocap[:, :13]
                target_g_j3d_mocap = target_g_j3d_mocap[:, :, :13]
                # mosh targets: local and global for beta and nobeta
                target_j3d_mosh_nobeta = target_j3d_mosh_nobeta[:, :13]
                target_g_j3d_mosh_nobeta = target_g_j3d_mosh_nobeta[:, :, :13]
                target_j3d_mosh_beta = target_j3d_mosh_beta[:, :13]
                target_g_j3d_mosh_beta = target_g_j3d_mosh_beta[:, :, :13]
                # predictions: local and global for beta and nobeta
                pred_j3d_nobeta = pred_j3d_nobeta[:, :13]
                pred_g_j3d_nobeta = pred_g_j3d_nobeta[:, :, :13]
                pred_j3d_beta = pred_j3d_beta[:, :13]
                pred_g_j3d_beta = pred_g_j3d_beta[:, :, :13]

            # pred keypoints
            self.evaluation_accumulators['pred_j3d'].append(pred_j3d_beta)
            self.evaluation_accumulators['pred_j3d_nobeta'].append(
                pred_j3d_nobeta)
            # target keypoints
            if 'gt_spin_joints3d' in hmr_cond['y']:
                self.evaluation_accumulators['target_j3d'].append(
                    target_j3d_mocap)
            self.evaluation_accumulators['target_j3d_mosh'].append(
                target_j3d_mosh_beta)
            self.evaluation_accumulators['target_j3d_mosh_nobeta'].append(
                target_j3d_mosh_nobeta)

            # pred global keypoints
            self.evaluation_accumulators['pred_g_j3d'].append(
                to_np(pred_g_j3d_beta))
            self.evaluation_accumulators['pred_g_j3d_nobeta'].append(
                to_np(pred_g_j3d_nobeta))
            # target global keypoints
            if 'gt_spin_joints3d' in hmr_cond['y']:
                self.evaluation_accumulators['target_g_j3d'].append(
                    to_np(target_g_j3d_mocap))
            self.evaluation_accumulators['target_g_j3d_mosh'].append(
                to_np(target_g_j3d_mosh_beta))
            self.evaluation_accumulators['target_g_j3d_mosh_nobeta'].append(
                to_np(target_g_j3d_mosh_nobeta))

            # metadata
            self.evaluation_accumulators['vid_name'].append(
                to_np(hmr_cond['y']['vid_name']))
            self.evaluation_accumulators['img_name'].append(
                hmr_cond['y']['img_name'])
            self.evaluation_accumulators['idxs'].append(to_np(idxs))
            self.evaluation_accumulators['pred_g_v3d'].append(
                to_np(g_pred_v3d_beta))
            self.evaluation_accumulators['pred_g_v3d_nobeta'].append(
                to_np(g_pred_v3d_nobeta))

            # if n_batches is not None: # debugging case
            self.evaluation_accumulators['pred_motions'].append(
                pred_motion.cpu())
            self.evaluation_accumulators['motions'].append(hmr_motion.cpu())
            self.evaluation_accumulators['target_theta'].append(
                hmr_cond['y']['theta'].cpu())
            self.evaluation_accumulators['orig_trans'].append(
                hmr_cond['y']['trans'].cpu())

            if self.args.db:
                break

        print('>>> Total Misses: ', miss_count)

    def eval_shortcut(self, epoch=0):
        # Just 1 dataset for now.
        eval_name, loader = list(self.eval_hmr_data_loader_dict.items())[0]
        data_name = loader.dataset.dataset
        data_rep = loader.dataset.data_rep
        print(" >> Eval_shortcut on: ", eval_name)

        # Just 1 batch for now.
        hmr_motion, hmr_cond = next(iter(loader))
        n, j, d, t = hmr_motion.shape
        video_features = hmr_cond['y']['features'].to(
            self.device)  # input vid features

        def _video2j3d(f):
            pred_motion = f(video_features)
            # correct the motion's orientation
            pred_motion_rot = remove_motion_rotation(pred_motion.clone(),
                                                     data_name=data_name,
                                                     data_rep=data_rep)
            # prediction kps without beta/shape
            pred_j3d_nobeta_dict = self.model.forward_kinematics(
                pred_motion_rot, None)
            pred_j3d_nobeta = to_np(pred_j3d_nobeta_dict['kp_c14_joints']
                                    )  # relative keypoints
            pred_j3d_trans_nobeta = to_np(
                pred_j3d_nobeta_dict['pred_trans'])  # translation
            pred_g_j3d_nobeta = pred_j3d_nobeta + pred_j3d_trans_nobeta  # global keypoints
            return pred_g_j3d_nobeta

        with torch.no_grad():
            f1 = lambda x: self.model.inference(x,
                                               use_dpm_solver=True,
                                               denoise_to_zero=True,
                                               steps=None,
                                               order=None)
            f2 = lambda x: self.model.inference(video_features,
                                               use_dpm_solver=True,
                                               denoise_to_zero=True,
                                               steps=1,
                                               order=1)
            f3 = lambda x: self.model.inference(video_features,
                                               use_dpm_solver=False,
                                               skip_timesteps=999)
            j1 = _video2j3d(f1)
            j2 = _video2j3d(f2)
            j3 = _video2j3d(f3)

        print('norm(j1-j2): ',torch.norm(to_tensor(j1 - j2)))
        print('norm(j1-j3): ',torch.norm(to_tensor(j1 - j3)))

    def validate_and_evaluate_all(self, epoch):
        for eval_name, eval_loader in self.eval_hmr_data_loader_dict.items():
            self.validate_hmr(eval_loader)
            self.evaluate_hmr_loop(epoch, eval_name)

    def validate_hmr(self,
                     loader,
                     use_dpm_solver=True,
                     n_batches=None,
                     do_vertices=False):
        """
        n_batches: restrict the number of batches to loop throgh. Used for debugging.

        """
        if n_batches == -1:
            n_batches = None

        if loader is None:
            print("Skipping eval ... no dataloader found")
            return

        print('  >> Accumulating predictionf for HMR evaluation.')

        # Refresh accumulator
        if self.evaluation_accumulators is not None:
            for k, v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        # loader = loader
        data_name = loader.dataset.dataset
        data_rep = loader.dataset.data_rep

        for i, eval_hmr_batch in tqdm(enumerate(loader), total=len(loader)):
            idxs = np.arange(len(loader)) + i * len(loader)
            if n_batches is not None and i >= n_batches:
                break

            hmr_motion, hmr_cond = eval_hmr_batch
            n, j, d, t = hmr_motion.shape

            ## predict motion, shape, and FK keypoints
            video_features = hmr_cond['y']['features'].to(
                self.device)  # input vid features
            hmr_motion = hmr_motion.to(self.device)  # gt motion

            with torch.no_grad():
                # predicted motion and shape
                if self.args.overwrite_dpm_steps > 0:
                    steps = self.args.overwrite_dpm_steps
                else:
                    steps = None

                if self.args.overwrite_dpm_order > 0:
                    order = self.args.overwrite_dpm_order
                else:
                    order = None

                pred_motion = self.model.inference(video_features,
                                                   use_dpm_solver=True,
                                                   denoise_to_zero=True,
                                                   steps=steps,
                                                   order=order)  # (N,25,6,T)
                pred_beta, _ = self.model.hmr_forward(video_features,
                                                      pred_motion)
                # correct the motion's orientation
                pred_motion_rot = remove_motion_rotation(pred_motion.clone(),
                                                         data_name=data_name,
                                                         data_rep=data_rep)
                # prediction kps without beta/shape
                pred_j3d_nobeta_dict = self.model.forward_kinematics(
                    pred_motion_rot, None)
                pred_j3d_nobeta = to_np(pred_j3d_nobeta_dict['kp_c14_joints']
                                        )  # relative keypoints
                pred_j3d_trans_nobeta = to_np(
                    pred_j3d_nobeta_dict['pred_trans'])  # translation
                pred_g_j3d_nobeta = pred_j3d_nobeta + pred_j3d_trans_nobeta  # global keypoints
                pred_g_j3d_nobeta = pred_g_j3d_nobeta.reshape(
                    n, t, 14, 3)  # reshape to include time for compute_g_mpjpe
                # prediction kps with beta/shape
                pred_j3d_beta_dict = self.model.forward_kinematics(
                    pred_motion_rot, pred_beta)
                pred_j3d_beta = to_np(
                    pred_j3d_beta_dict['kp_c14_joints'])  # relative keypoints
                pred_j3d_trans_beta = to_np(
                    pred_j3d_beta_dict['pred_trans'])  # translation
                pred_g_j3d_beta = pred_j3d_beta + pred_j3d_trans_beta  # global keypoints
                pred_g_j3d_beta = pred_g_j3d_beta.reshape(
                    n, t, 14, 3)  # needed for compute_g_mpjpe

            ## global vertices for rendering
            if do_vertices:
                pred_v3d_nobeta = to_np(pred_j3d_nobeta_dict['pred_vertices'])
                g_pred_v3d_nobeta = pred_v3d_nobeta + to_np(
                    pred_j3d_nobeta_dict['pred_trans'])
                g_pred_v3d_nobeta = g_pred_v3d_nobeta.reshape(n, t, -1, 3)
                # global vertices with beta
                pred_v3d_beta = to_np(pred_j3d_beta_dict['pred_vertices'])
                g_pred_v3d_beta = pred_v3d_nobeta + to_np(
                    pred_j3d_beta_dict['pred_trans'])
                g_pred_v3d_beta = g_pred_v3d_beta.reshape(n, t, -1, 3)

            ## mocap target keypoints
            if 'gt_spin_joints3d' in hmr_cond['y']:
                target_j3d_mocap = to_np(
                    hmr_cond['y']['gt_spin_joints3d'])[..., :3]  # (N,T,49,3)
                n, t, j, _ = target_j3d_mocap.shape
                target_j3d_mocap = convert_kps(
                    target_j3d_mocap.reshape(n * t, j, 3), 'spin', 'common')
                target_g_j3d_mocap = target_j3d_mocap.reshape(
                    n, t, 14,
                    3)  # global kps are the same data, but reshaped (n,t,14,3)

            ## mosh target keypoints: both with and without gt beta
            with torch.no_grad():
                # gt motion was previously rotated - remove it
                hmr_motion_rot = remove_motion_rotation(hmr_motion.clone(),
                                                        data_name=data_name,
                                                        data_rep=data_rep)
                # kps without beta: local and global
                target_j3d_mosh_nobeta_dict = self.model.forward_kinematics(
                    hmr_motion_rot, None)
                target_j3d_mosh_nobeta = to_np(
                    target_j3d_mosh_nobeta_dict['kp_c14_joints'])
                target_j3d_mosh_nobeta += to_np(
                    target_j3d_mosh_nobeta_dict['pred_trans'])
                target_g_j3d_mosh_nobeta = target_j3d_mosh_nobeta.reshape(
                    n, t, 14,
                    3)  # global kps are the same data, but reshaped (n,t,14,3)
                # kps without beta: local and global
                target_beta = hmr_cond['y']['theta'][..., -10:].to(
                    self.device).float()  # (N,T,10)
                target_j3d_mosh_beta_dict = self.model.forward_kinematics(
                    hmr_motion_rot, target_beta)
                target_j3d_mosh_beta = to_np(
                    target_j3d_mosh_beta_dict['kp_c14_joints'])
                target_j3d_mosh_beta += to_np(
                    target_j3d_mosh_beta_dict['pred_trans'])
                target_g_j3d_mosh_beta = target_j3d_mosh_beta.reshape(
                    n, t, 14,
                    3)  # global kps are the same data, but reshaped (n,t,14,3)

            ## for common14 joints, h36m doesn't have the last one, so remove them
            if loader.dataset.dataset == 'h36m':
                # mocap targets: local and global
                if 'gt_spin_joints3d' in hmr_cond['y']:
                    target_j3d_mocap = target_j3d_mocap[:, :13]
                    target_g_j3d_mocap = target_g_j3d_mocap[:, :, :13]
                # mosh targets: local and global for beta and nobeta
                target_j3d_mosh_nobeta = target_j3d_mosh_nobeta[:, :13]
                target_g_j3d_mosh_nobeta = target_g_j3d_mosh_nobeta[:, :, :13]
                target_j3d_mosh_beta = target_j3d_mosh_beta[:, :13]
                target_g_j3d_mosh_beta = target_g_j3d_mosh_beta[:, :, :13]
                # predictions: local and global for beta and nobeta
                pred_j3d_nobeta = pred_j3d_nobeta[:, :13]
                pred_g_j3d_nobeta = pred_g_j3d_nobeta[:, :, :13]
                pred_j3d_beta = pred_j3d_beta[:, :13]
                pred_g_j3d_beta = pred_g_j3d_beta[:, :, :13]

            # pred keypoints
            self.evaluation_accumulators['pred_j3d'].append(pred_j3d_beta)
            self.evaluation_accumulators['pred_j3d_nobeta'].append(
                pred_j3d_nobeta)
            # target keypoints
            if 'gt_spin_joints3d' in hmr_cond['y']:
                self.evaluation_accumulators['target_j3d'].append(
                    target_j3d_mocap)
            self.evaluation_accumulators['target_j3d_mosh'].append(
                target_j3d_mosh_beta)
            self.evaluation_accumulators['target_j3d_mosh_nobeta'].append(
                target_j3d_mosh_nobeta)

            # pred global keypoints
            self.evaluation_accumulators['pred_g_j3d'].append(
                to_np(pred_g_j3d_beta))
            self.evaluation_accumulators['pred_g_j3d_nobeta'].append(
                to_np(pred_g_j3d_nobeta))
            # target global keypoints
            if 'gt_spin_joints3d' in hmr_cond['y']:
                self.evaluation_accumulators['target_g_j3d'].append(
                    to_np(target_g_j3d_mocap))
            self.evaluation_accumulators['target_g_j3d_mosh'].append(
                to_np(target_g_j3d_mosh_beta))
            self.evaluation_accumulators['target_g_j3d_mosh_nobeta'].append(
                to_np(target_g_j3d_mosh_nobeta))

            # metadata
            self.evaluation_accumulators['vid_name'].append(
                to_np(hmr_cond['y']['vid_name']))
            self.evaluation_accumulators['img_name'].append(
                hmr_cond['y']['img_name'])
            self.evaluation_accumulators['idxs'].append(to_np(idxs))

            if do_vertices:
                self.evaluation_accumulators['pred_g_v3d'].append(
                    to_np(g_pred_v3d_beta))
                self.evaluation_accumulators['pred_g_v3d_nobeta'].append(
                    to_np(g_pred_v3d_nobeta))

            # if n_batches is not None: # debugging case
            self.evaluation_accumulators['pred_motions'].append(
                pred_motion.cpu())
            self.evaluation_accumulators['motions'].append(hmr_motion.cpu())
            self.evaluation_accumulators['target_theta'].append(
                hmr_cond['y']['theta'].cpu())
            self.evaluation_accumulators['orig_trans'].append(
                hmr_cond['y']['trans'].cpu())

            if self.args.db:
                break

            # pred_verts = preds[-1]['verts'].view(-1, 6890,
            #                                      3).cpu().numpy()
            # target_theta = target['theta'].view(-1, 85).cpu().numpy()

            # self.evaluation_accumulators['pred_verts'].append(
            #     pred_verts)
            # self.evaluation_accumulators['target_theta'].append(
            #     target_theta)

    def _evaluate_hmr_core(self,
                           evaluation_accumulators,
                           epoch,
                           do_mosh=True,
                           nemo=False):
        # prediction and target kps: predictions and mosh targets are computed with and without beta.
        pred_j3ds = evaluation_accumulators['pred_j3d']
        pred_j3d_nobeta = evaluation_accumulators['pred_j3d_nobeta']
        target_j3ds = evaluation_accumulators['target_j3d']
        if do_mosh:
            target_j3ds_mosh = evaluation_accumulators['target_j3d_mosh']
            target_j3ds_mosh_nobeta = evaluation_accumulators[
                'target_j3d_mosh_nobeta']

        # MPJPE & PA-MPJPE
        if len(target_j3ds) == 0:
            errors = errors_pa = errors_nobeta = errors_pa_nobeta = -1
        else:
            errors, errors_pa = compute_mpjpe(pred_j3ds, target_j3ds)
            errors_nobeta, errors_pa_nobeta = compute_mpjpe(
                pred_j3d_nobeta, target_j3ds)  # the prediction has beta
        if do_mosh:
            errors_mosh, errors_pa_mosh = compute_mpjpe(
                pred_j3ds, target_j3ds_mosh)
            errors_mosh_nobeta, errors_pa_mosh_nobeta = compute_mpjpe(
                pred_j3d_nobeta, target_j3ds_mosh_nobeta)

        # G-MPJPE
        if len(target_j3ds) == 0:
            g_mpjpe = g_mpjpe_nobeta = g_vel = g_acc = -1
        else:
            g_mpjpe = compute_g_mpjpe(evaluation_accumulators['pred_g_j3d'],
                                      evaluation_accumulators['target_g_j3d'])
            g_mpjpe_nobeta = compute_g_mpjpe(
                evaluation_accumulators['pred_g_j3d_nobeta'],
                evaluation_accumulators['target_g_j3d'])
            g_vel = compute_error_g_vel(
                evaluation_accumulators['pred_g_j3d'],
                evaluation_accumulators['target_g_j3d'])
            g_acc = compute_error_g_acc(
                evaluation_accumulators['pred_g_j3d'],
                evaluation_accumulators['target_g_j3d'])

        if do_mosh:
            g_mpjpe_mosh = compute_g_mpjpe(
                evaluation_accumulators['pred_g_j3d'],
                evaluation_accumulators['target_g_j3d_mosh'])
            nemo_mpjpe_mosh = compute_nemo_mpjpe(
                evaluation_accumulators['pred_g_j3d'],
                evaluation_accumulators['target_g_j3d_mosh'])
            g_mpjpe_mosh_nobeta = compute_g_mpjpe(
                evaluation_accumulators['pred_g_j3d_nobeta'],
                evaluation_accumulators['target_g_j3d_mosh_nobeta'])
            g_vel_mosh = compute_error_g_vel(
                evaluation_accumulators['pred_g_j3d'],
                evaluation_accumulators['target_g_j3d_mosh'])
            g_acc_mosh = compute_error_g_acc(
                evaluation_accumulators['pred_g_j3d'],
                evaluation_accumulators['target_g_j3d_mosh'])

        m2mm = 1000

        do_vertices = True if len(
            evaluation_accumulators['pred_g_v3d']) > 0 else False
        # (G-)VPE
        g_pred_verts = to_np(evaluation_accumulators['pred_g_v3d'])
        g_target_theta = to_np(evaluation_accumulators['target_theta'].float())
        target_trans = to_np(evaluation_accumulators['orig_trans'])
        target_theta = to_np(evaluation_accumulators['target_theta'].reshape(
            -1, 85).float())

        if do_vertices:
            pred_verts = to_np(evaluation_accumulators['pred_g_v3d'].reshape(
                -1, 6890, 3))

            pve = np.mean(
                compute_error_verts(target_theta=target_theta,
                                    pred_verts=pred_verts)) * m2mm

            gpve = np.mean(
                compute_error_g_verts(target_theta=g_target_theta,
                                      pred_verts=g_pred_verts,
                                      target_trans=target_trans)) * m2mm

        else:
            pve = -1
            gpve = -1
        nemo_pve = -1
        if nemo and do_vertices:
            g_target_verts = f_target_vert(g_target_theta, target_trans)
            nemo_pve = np.mean(compute_nemo_mpjpe(g_pred_verts,
                                                  g_target_verts)) * m2mm
        m2mm = 1000

        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        if len(target_j3ds) == 0:
            accel_err = -1
        else:
            accel_err = np.mean(
                compute_error_accel(joints_pred=pred_j3ds,
                                    joints_gt=target_j3ds)) * m2mm

        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm
        g_mpjpe = np.mean(g_mpjpe) * m2mm

        mpjpe_nobeta = np.mean(errors_nobeta) * m2mm
        pa_mpjpe_nobeta = np.mean(errors_pa_nobeta) * m2mm
        g_mpjpe_nobeta = np.mean(g_mpjpe_nobeta) * m2mm

        # G-Vel/Acc
        g_vel = np.mean(g_vel) * m2mm
        g_acc = np.mean(g_acc) * m2mm

        if do_mosh:
            mosh_mpjpe = np.mean(errors_mosh) * m2mm
            mosh_pa_mpjpe = np.mean(errors_pa_mosh) * m2mm
            mosh_g_mpjpe = np.mean(g_mpjpe_mosh) * m2mm

            mosh_mpjpe_nobeta = np.mean(errors_mosh_nobeta) * m2mm
            mosh_pa_mpjpe_nobeta = np.mean(errors_pa_mosh_nobeta) * m2mm
            mosh_g_mpjpe_nobeta = np.mean(g_mpjpe_mosh_nobeta) * m2mm
            nemo_mpjpe_mosh = np.mean(nemo_mpjpe_mosh) * m2mm
            g_vel_mosh = np.mean(g_vel_mosh) * m2mm
            g_acc_mosh = np.mean(g_acc_mosh) * m2mm

        eval_dict = {
            'epoch': epoch,
            'g_mpjpe': g_mpjpe,
            'mpjpe': mpjpe,
            'pa_mpjpe': pa_mpjpe,
            'pve': pve,
            'gpve': gpve,
            'nemo_pve': nemo_pve,
            'g_mpjpe_nobeta': g_mpjpe_nobeta,
            'mpjpe_nobeta': mpjpe_nobeta,
            'pa_mpjpe_nobeta': pa_mpjpe_nobeta,
            'g_vel': g_vel,
            'g_acc': g_acc
        }
        if do_mosh:
            eval_dict.update({
                'mosh_g_mpjpe': mosh_g_mpjpe,
                'mosh_mpjpe': mosh_mpjpe,
                'nemo_mpjpe_mosh': nemo_mpjpe_mosh,
                'mosh_pa_mpjpe': mosh_pa_mpjpe,
                'mosh_g_mpjpe_nobeta': mosh_g_mpjpe_nobeta,
                'mosh_mpjpe_nobeta': mosh_mpjpe_nobeta,
                'mosh_pa_mpjpe_nobeta': mosh_pa_mpjpe_nobeta,
                'mosh_g_vel': g_vel_mosh,
                'mosh_g_acc': g_acc_mosh
            })
        return eval_dict

    def evaluate_hmr_loop(self, epoch, eval_name='', do_mosh=True, nemo=False):
        """
        A loop version of `evaluate_hmr` to avoid the np.vstack in the beginning as it causes CPU OOM.
        """

        print('  >> [Evaluating HMR]')
        Nb = len(self.evaluation_accumulators['pred_g_j3d'])
        Bs = len(self.evaluation_accumulators['pred_g_j3d'][0])

        all_eval_dict = []
        for batch_idx in tqdm(range(Nb)):
            # Create an accumulator for just 1 batch
            cur_eval_accu = {}
            for k, v in self.evaluation_accumulators.items():
                cur_eval_accu[k] = v[batch_idx] if len(v) > 0 else []

            eval_dict = self._evaluate_hmr_core(cur_eval_accu,
                                                epoch,
                                                do_mosh,
                                                nemo=nemo)
            eval_dict['bs'] = len(cur_eval_accu['pred_g_j3d'])
            all_eval_dict.append(eval_dict)

        eval_dict = defaultdict(int)

        # Compute weighted average
        # 1. Add all numbers up
        for k in all_eval_dict[0].keys():
            for ed in all_eval_dict:
                eval_dict[k] += ed[k] * ed['bs'] if k != 'bs' else ed[k]

        # 2. Divide by N
        for k in eval_dict:
            eval_dict[k] /= eval_dict['bs']

        self.print_and_save_eval_dict(eval_dict, epoch, eval_name)

    def evaluate_hmr(self, epoch, eval_name='', do_mosh=True):
        # if self.eval_hmr_data_loader is None:
        #     return
        print('  >> [Evaluating HMR]')
        for k, v in self.evaluation_accumulators.items():
            if len(self.evaluation_accumulators[k]) > 0:
                self.evaluation_accumulators[k] = np.vstack(v)

        eval_dict = self._evaluate_hmr_core(self.evaluation_accumulators,
                                            epoch, do_mosh)

        self.print_and_save_eval_dict(epoch, eval_name)

    def print_and_save_eval_dict(self, eval_dict, epoch, eval_name=''):
        if eval_name != '':
            new_eval_dict = {}
            for k, v in eval_dict.items():
                new_eval_dict[f"{eval_name}.{k}"] = v
            eval_dict = new_eval_dict

        # Print evaluation metrics
        print('Epoch: ', epoch)
        for k, v in eval_dict.items():
            print(f"  `{k}`:", v)

        # Log Eval
        out_name = 'hmr_eval.csv' if eval_name == '' else f'fhmr_eval.{eval_name}.csv'
        log_dict(eval_dict, osp.join(self.save_dir, out_name))

    def run_loop(self):

        for epoch in range(self.num_epochs):
            # Evaluate on the HMR task
            if (epoch % self.args.eval_hmr_every == 0):
                # if epoch!=0:
                self.model.eval()
                # self.validate_hmr()
                # self.evaluate_hmr(epoch)
                self.validate_and_evaluate_all(epoch)
                self.model.train()

            print(f'Starting epoch {epoch}')
            # for motion, cond in tqdm(self.data):
            for iter_i in tqdm(range(self.num_iters_per_epoch)):

                # Load datas
                hmr_batch = motion_batch = None
                if self.train_hmr_data_iter_dict:
                    all_hmr_motion, all_hmr_cond = [], []
                    for name, train_hmr_data_iter in self.train_hmr_data_iter_dict.items(
                    ):
                        try:
                            hmr_batch = next(train_hmr_data_iter)
                        except StopIteration:
                            train_hmr_data_iter = iter(
                                self.train_hmr_data_loader_dict[name])
                            self.train_hmr_data_iter_dict[
                                name] = train_hmr_data_iter
                            hmr_batch = next(train_hmr_data_iter)
                        hmr_motion, hmr_cond = hmr_batch
                        all_hmr_motion.append(hmr_motion)
                        all_hmr_cond.append(hmr_cond)
                    hmr_motion = torch.cat(all_hmr_motion, 0)
                    hmr_cond = {'y': defaultdict(list)}
                    # Accumulate all values
                    for key in all_hmr_cond[0]['y'].keys():
                        for c_hmr_cond in all_hmr_cond:
                            if key in c_hmr_cond['y'].keys():
                                hmr_cond['y'][key].append(c_hmr_cond['y'][key])
                    # Merge all values
                    for key in hmr_cond['y'].keys():
                        if isinstance(hmr_cond['y'][key],
                                      torch.Tensor) or isinstance(
                                          hmr_cond['y'][key][0], torch.Tensor):
                            hmr_cond['y'][key] = torch.cat(
                                hmr_cond['y'][key], 0)
                        elif isinstance(hmr_cond['y'][key],
                                        np.ndarray) or isinstance(
                                            hmr_cond['y'][key][0], np.ndarray):
                            hmr_cond['y'][key] = np.vstack(hmr_cond['y'][key])
                        elif isinstance(hmr_cond['y'][key], list):
                            hmr_cond['y'][key] = sum(hmr_cond['y'][key], [])
                if self.train_motion_data_iter_dict:
                    all_motion_motion, all_motion_cond = [], []
                    for name, train_motion_data_iter in self.train_motion_data_iter_dict.items(
                    ):
                        try:
                            motion_batch = next(train_motion_data_iter)
                        except StopIteration:
                            train_motion_data_iter = iter(
                                self.train_motion_data_loader_dict[name])
                            self.train_motion_data_iter_dict[
                                name] = train_motion_data_iter
                            motion_batch = next(train_motion_data_iter)
                        motion_motion, motion_cond = motion_batch
                        all_motion_motion.append(motion_motion)
                        all_motion_cond.append(motion_cond)
                    motion_motion = torch.cat(all_motion_motion, 0)
                    motion_cond = {'y': defaultdict(list)}
                    
                    # Accumulate all values
                    for key in all_motion_cond[0]['y'].keys():
                        for c_motion_cond in all_motion_cond:
                            motion_cond['y'][key].append(
                                c_motion_cond['y'][key])
                    # Merge all values
                    for key in motion_cond['y'].keys():
                        if isinstance(motion_cond['y'][key],
                                      torch.Tensor) or isinstance(
                                          motion_cond['y'][key][0],
                                          torch.Tensor):
                            motion_cond['y'][key] = torch.cat(
                                motion_cond['y'][key], 0)
                        elif isinstance(motion_cond['y'][key],
                                        np.ndarray) or isinstance(
                                            motion_cond['y'][key][0],
                                            np.ndarray):
                            motion_cond['y'][key] = np.vstack(
                                motion_cond['y'][key])
                        elif isinstance(motion_cond['y'][key], list):
                            motion_cond['y'][key] = sum(
                                motion_cond['y'][key], [])

                    # Generate dummy video features for motion only batch
                    N, J, _, T = motion_motion.shape
                    motion_cond['y']['features'] = torch.zeros(
                        (N, T, 2048), dtype=motion_motion.dtype)

                if hmr_batch is not None and motion_batch is not None:
                    # Mix HMR batch and Motion batch
                    motion = torch.cat([hmr_motion, motion_motion], 0)
                    cond = {'y': {}}
                    for key in motion_cond['y'].keys():
                        if key in ['action_text', 'vid_name', 'img_name']:
                            continue
                        val1 = hmr_cond['y'][key]
                        val2 = motion_cond['y'][key]
                        cond['y'][key] = torch.cat(
                            [to_tensor(val1), to_tensor(val2)], 0)
                else:
                    if hmr_batch is not None:
                        motion = hmr_motion
                        cond = hmr_cond
                    else:
                        motion = motion_motion
                        cond = motion_cond

                if not (not self.lr_anneal_steps or
                        self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond['y'].items()
                }

                # Pass thru the MDM component
                self.run_step(motion, cond)

                # run hmr module forward pass (unless there is not hmr train loader)
                if self.train_hmr_data_loader_dict and (
                        self.args.update_hmr_every
                        >= 0) and (iter_i % self.args.update_hmr_every == 0):
                    raise  # OBSOLETE
                    video_features = hmr_cond['y']['features'].cuda()
                    # with torch.no_grad():
                    #     pred_slv_motion = self.model.inference(
                    #         video_features, use_dpm_solver=True, steps=5)
                    # pred_hmr_motion = self.model.run_pitch_correction(
                    #     video_features, pred_slv_motion)

                    hmr_gen_outputs = self.model.inference_hmr(
                        video_features, use_dpm_solver=True, steps=5)

                    N, T = hmr_cond['y']['joints3D'].shape[:2]
                    data_3d = {
                        'kp_3d': to_tensor(hmr_cond['y']['joints3D']),
                        'kp_2d': to_tensor(hmr_cond['y']['kp_2d']),
                        'theta': to_tensor(hmr_cond['y']['theta']),
                        'w_3d': torch.ones((N, T)).cuda(),
                        'w_smpl': torch.ones((N, T)).cuda()
                    }

                    gen_loss, loss_dict = self.hmr_loss(hmr_gen_outputs,
                                                        data_2d=None,
                                                        data_3d=data_3d)
                    self.hmr_opt.zero_grad()
                    gen_loss.backward()
                    self.hmr_opt.step()

                if self.step % self.log_interval == 0:
                    for k, v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(
                                self.step + self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k,
                                value=v,
                                iteration=self.step,
                                group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST",
                                      "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(
                self.save_dir,
                f'eval_humanml_{(self.step + self.resume_step):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper,
                self.eval_gt_data,
                self.eval_data,
                log_file,
                replication_times=self.args.eval_rep_times,
                diversity_times=diversity_times,
                mm_num_times=mm_num_times,
                run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(
                            name=f'top{i + 1}_' + k,
                            value=v[i],
                            iteration=self.step + self.resume_step,
                            group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k,
                                                      value=v,
                                                      iteration=self.step +
                                                      self.resume_step,
                                                      group_name='Eval')

        elif self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times,
                                        num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size,
                                        device=self.device,
                                        guidance_param=1,
                                        dataset=self.dataset,
                                        unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(
                                            self.save_dir,
                                            self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(
                eval_args,
                model=self.model,
                diffusion=self.diffusion,
                data=self.data.dataset)
            print(
                f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}'
            )
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name='Eval')
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0],
                                                      dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset='h36m')


            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {
                k: v * weights
                for k, v in losses.items()
            })
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples",
                     (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):

        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [
                e for e in state_dict.keys() if e.startswith('clip_model.')
            ]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
                bf.join(self.save_dir,
                        f"opt{(self.step+self.resume_step):09d}.pt"),
                "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(),
                                   values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
