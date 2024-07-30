"""
Based on mdm/sample/generate.py

Generate samples conditioned on video features. This means running GHMR inference. 

python -m mdm.sample.generate_vid_cond \
                --condition_dataset nemomocap \
                --condition_split train \
                --model_path /oak/stanford/groups/syyeung/jmhb/mdm_save_directory/save/20230227_trainemp_exp14-0-h36m_only/000000/model000090000.pt \
                --output_dir tmp \
                --seed 0 \
                --num_samples 32 \
                --num_repetitions 2 

Required args: 
    --model_path diffusion model to sample from. 
Optional args:     
    --condition_dataset one of 'h36m','3dpw'. If empty, defaults to the dataset the model was trained on. Note that args.dataset will be the same as model dataset ... this param is used for some other config (see the code)
    --condition_split. One of 'train','test'. Defaults to 'test'.  
    --output_dir. Where to put result images. if not specified, put in same folder as --model_path
    --num_samples default 10
    --num_repetitions default 3
"""
from mdm.utils.fixseed import fixseed
import os
import argparse
import json
import numpy as np
import torch
from mdm.utils.parser_util import generate_args, train_args
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from  gthmr.emp_train.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
from mdm.utils.model_util import create_emp_model_and_diffusion
import shutil
from mdm.data_loaders.tensors import collate
from gthmr.lib.utils.mdm_utils import viz_motions
from VIBE.lib.dataset.vibe_dataset import rotate_about_D
import ipdb

def main():
    print("Generating video-conditioned samples and ")
    args = generate_args()
    args.unconstrained=False
    print(f"    Condition dataset [{args.condition_dataset}]")
    print(f"                split [{args.condition_split}]")
    
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)

    if args.output_dir=="":
        args.output_dir = os.path.dirname(args.model_path)


    print(f"Out path: [{out_path}]")

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print("Creating model and diffusion...")
    """
    if args.emp:
        model, diffusion = create_emp_model_and_diffusion(args, None)
    else:
        model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    """
    if args.model_path != "":
        args.pretrained_model_path = args.model_path
        pretrained_model_path = args.pretrained_model_path
        print(f"Loading pretrained model [{args.pretrained_model_path}]")
        path_model_args = os.path.join(
            os.path.dirname(args.pretrained_model_path), "args.json")
        print("Loading ", path_model_args)
        if not os.path.exists(path_model_args):
            raise ValueError(f"Model path [{args.pretrained_model_path}] must be in the same" \
                            "directory as its model args file: [args.json]")
        with open(path_model_args, 'r') as f:
            args_pretrained_model = argparse.Namespace(**json.load(f))

        # check that the data_rep specifi
        args.data_rep = args_pretrained_model.data_rep

        # we will make `args_pretrained_model` the global args. But we want to
        # keep some config from `args` the same, so copy those values from
        # `args` to `args_pretrained_model`
        for k in [
                'condition_dataset', 'condition_split', 'num_samples',
                'num_repetitions','output_dir', 'condition_dataset','seed', 
                'guidance_param', 'unconstrained', 'batch_size'
        ]:
            setattr(args_pretrained_model, k, getattr(args, k))

        # now make the pretrained config the global config
        args = args_pretrained_model

        # args.video_arch_experiment = 1
        # load the pretrained model
        model, diffusion = create_emp_model_and_diffusion(
            args_pretrained_model, None)
        state_dict = torch.load(pretrained_model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

    print(f'Loading dataset for conditioning features {args.condition_dataset}...')
    data = load_dataset(args, max_frames, n_frames, split=args.condition_split, 
        data_rep=args.data_rep)

    total_num_samples = args.num_samples * args.num_repetitions

    # model.cond_mode="no_cond"
    if args.guidance_param != 1 and model.cond_mode!="no_cond": # only do cfgsampler for conditional models
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    all_motions = []
    all_lengths = []
    all_text = []

    # first show the motion for the sampled condition
    def postprocess(sample, model, data,):

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)
        return sample

    # get data: 
    #   the `samples_vid_gt` are the ground truth motions 
    #   the `model_kwargs` have the video features used for conditioning. 
    samples_vid_gt, model_kwargs = next(iter(data))
    # get the ground trouth motion for the first column: we want the output samples to match this motion
    # 
    samples_vid_gt_processed = postprocess(samples_vid_gt.to(dist_util.dev()), model, data,)
    all_motions.append(samples_vid_gt_processed.cpu().numpy())
    all_text += [f'GT cond motion {args.condition_dataset }']*len(samples_vid_gt_processed)
    all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        ZERO_FEATURES=False
        if ZERO_FEATURES: 
            model_kwargs['y']['features'] = torch.zeros_like(model_kwargs['y']['features'])

        sample_fn = diffusion.p_sample_loop

        with torch.no_grad():
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        
        sample = postprocess(sample, model, data)
        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            all_text += [f'conditioned sample {rep_i}']* args.num_samples

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    nrows = len(all_motions[0])
    ncols = args.num_repetitions+1  # bc we added the ground truth
    all_motions = np.vstack(all_motions)

    viz_motions(nrows, ncols, out_path, all_motions, dataset=None, all_text=all_text)
    ipdb.set_trace()

    # viz_motions(nrows, ncols, out_path, all_motions, dataset=data.dataset.dataname, all_text=None)
    print(f'[Done] Results are at [{os.path.abspath(out_path)}]')

def load_dataset(args, max_frames, n_frames, split, data_rep):
    data = get_dataset_loader(name=args.condition_dataset,
                              batch_size=args.num_samples,
                              num_frames=max_frames,
                              split=args.condition_split,
                              data_rep=data_rep,
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
