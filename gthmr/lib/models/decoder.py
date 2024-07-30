import os
from nemo.utils.exp_utils import create_latest_child_dir
import torch.nn as nn
from torch.autograd import Variable
from nemo.utils import to_np, to_tensor

from mdm.utils.fixseed import fixseed
import os
import numpy as np
import torch
from mdm.utils.parser_util import generate_parser, parse_and_load_from_model
from gthmr.lib.utils.mdm_utils import get_args_from_model_path
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.data_loaders.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from mdm.data_loaders.tensors import collate
from dpm_solver import DPM_Solver, model_wrapper, NoiseScheduleVP
import ipdb

default_dpm_solver_args = {
    'algorithm_type': "dpmsolver++",
    'method': 'multistep',
    'order': 3,
    'steps': 20,
}


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


class MDMDecoder:
    """docstring for MDMDecoder"""

    def __init__(self,
                 model_path,
                 use_dpm_solver=False,
                 dpm_solver_args=default_dpm_solver_args):
        super(MDMDecoder, self).__init__()
        self.model_path = model_path
        self.use_dpm_solver = use_dpm_solver
        self.dpm_solver_args = dpm_solver_args
        args = get_args_from_model_path(model_path)
        self.mdm_args = args
        args.batch_size = args.num_samples
        print("MDM args: ")
        print(args)
        print('Loading dataset for initialization')
        n_frames = 196
        max_frames = 196
        data = load_dataset(args, max_frames, n_frames)

        # stuff = next(data)
        # ipdb.set_trace()

        print("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, data)

        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

        model.to(args.device)
        model.eval()  # disable random masking

        model.cond_mode = 'no_cond'

        if self.use_dpm_solver:
            noise_schedule = NoiseScheduleVP(betas=to_tensor(diffusion.betas))
            wrapped_model_fn = model_wrapper(self.model_fn,
                                             noise_schedule,
                                             model_type='x_start')
            self.dpm_solver = DPM_Solver(
                wrapped_model_fn,
                noise_schedule,
                algorithm_type=self.dpm_solver_args['algorithm_type'])

        self.model = model
        self.diffusion = diffusion

    # Recover XYZ *positions* from HumanML3D vector representation
    def postprocess(self, sample):
        model = self.model
        if model.data_rep == 'hml_vec':
            raise  # see original MDM code
        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'
                                                       ] else model.data_rep
        rot2xyz_mask = None
        sample = model.rot2xyz(x=sample,
                               mask=rot2xyz_mask,
                               pose_rep=rot2xyz_pose_rep,
                               glob=True,
                               translation=True,
                               jointstype='smpl',
                               vertstrans=True,
                               betas=None,
                               beta=0,
                               glob_rot=None,
                               get_rotations_back=False)
        return sample

    def decode(self, noise):
        """
        Input
            noise -- shape is (args.batch_size, model.njoints, model.nfeats, n_frames)
        Return 
            sample -- 
        """
        if self.use_dpm_solver:
            return self.dpm_solver_decode(noise)
        else:
            return self.mdm_decode(noise)

    def mdm_decode(self, noise):
        diffusion, model = self.diffusion, self.model

        noise_shape = noise.shape
        sample_fn = diffusion.p_sample_loop
        # sample = sample_fn(
        #     model,
        #     noise_shape,
        #     clip_denoised=False,
        #     model_kwargs=None,
        #     skip_timesteps=999,
        #     init_image=None,
        #     progress=False,
        #     cond_fn_with_grad=False,  # Enable gradients
        #     dump_steps=None,
        #     noise=noise,
        #     const_noise=False,  # this uses the same noise for all samples.
        #     const_t_noise=True,
        #     # indices=list(np.arange(99, 500, 100)[::-1]) + [0],
        # )  # shape (N,J,D,T) with J=25 and D=6

        N = noise.shape[0]
        # sample = diffusion.p_mean_variance(model,
        #                                    noise,
        #                                    torch.zeros((N)).long().cuda(),
        #                                    clip_denoised=False)['pred_xstart']

        sample = model(noise, torch.zeros((N)).long().cuda(), **{'y':{}})
        # sample = noise

        # with torch.no_grad():
        # sample = sample_fn(
        #     model,
        #     noise_shape,
        #     clip_denoised=False,
        #     model_kwargs=None,
        #     skip_timesteps=0,
        #     init_image=None,
        #     progress=False,
        #     cond_fn_with_grad=False,  # Enable gradients
        #     dump_steps=None,
        #     noise=noise,
        #     const_noise=False,  # this uses the same noise for all samples.
        #     const_t_noise=False,
        #     # indices=list(np.arange(99, 500, 100)[::-1]) + [0],
        # )
        return sample

    def model_fn(self, x, t):
        model = self.model
        diffusion = self.diffusion
        return diffusion.p_mean_variance(model, x, t,
                                         clip_denoised=False)['pred_xstart']

    def dpm_solver_decode(self, noise):
        dpm_solver = self.dpm_solver
        sample = dpm_solver.sample(
            x=noise,
            method=self.dpm_solver_args['method'],
            order=self.dpm_solver_args['order'],
            steps=self.dpm_solver_args['steps'],
            skip_type="logSNR",
            denoise_to_zero=False,
            no_grad=False
            # optional: skip_type='time_uniform' or 'logSNR' or 'time_quadratic',
        )
        return sample


if __name__ == "__main__":
    model_path = 'mdm/save/unconstrained/model000450000.pt'
    decoder = MDMDecoder(model_path, use_dpm_solver=True)

    batch_size = 10
    n_frames = 16
    noise_shape = (batch_size, decoder.model.njoints, decoder.model.nfeats,
                   n_frames)  # (N, 25, 6, T)
    noise = Variable(torch.randn(noise_shape, device='cuda:0'),
                     requires_grad=True)
    sample = decoder.decode(noise)
    print(sample.shape)
