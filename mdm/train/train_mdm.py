# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import ipdb
from mdm.utils.fixseed import fixseed
from mdm.utils.parser_util import train_args
from mdm.utils import dist_util
from mdm.train.training_loop import TrainLoop
from mdm.data_loaders.get_data import get_dataset_loader
from mdm.utils.model_util import create_model_and_diffusion
from mdm.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from nemo.utils.exp_utils import create_latest_child_dir


def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')

    args.save_dir = create_latest_child_dir(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    dist_util.setup_dist(args.device)

    print(f"Creating data loader for dataset {args.dataset}, num_workers {args.num_workers}")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        foot_vel_threshold=args.foot_vel_threshold, num_frames=args.num_frames, rotation_augmentation=args.rotation_augmentation,
        data_rep=args.data_rep, no_motion_augmentation=args.no_motion_augmentation, split=args.split) 

    print("Dataset size ", len(data.dataset))

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
