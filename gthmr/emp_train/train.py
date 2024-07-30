"""
Testing script:
    python -m gthmr.emp_train.train \
     --save_dir gthmr/results/20230306_masked_training/tmp \
     --data_config_path gthmr/emp_train/config/data/dev_rot6dFcShapeAxyz.yml \
     --pretrained_model_path /oak/stanford/groups/syyeung/jmhb/mdm_save_directory/save/20230304_trainemp_exp20-0-h36m_rot6dFcShapeAxyz-h36m_rot6dFcShapeAxyz/000004/model000260000.pt \
     --data_config_path gthmr/emp_train/config/data/dev_rot6dFcShapeAxyz.yml  \
     --feature_mask_training_exp 1
"""

import os
import json
import ipdb
import torch
from mdm.utils.fixseed import fixseed
from mdm.utils.parser_util import train_emp_args
from mdm.utils import dist_util
from gthmr.emp_train.training_loop import TrainLoop
from gthmr.emp_train.get_data import get_dataset_loader_dict_new2
from mdm.utils.model_util import create_emp_model_and_diffusion, load_model_wo_clip
from mdm.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from nemo.utils.exp_utils import create_latest_child_dir
import argparse
import yaml


def main():
    args = train_emp_args()
    fixseed(args.seed)

    # Load and parse data config
    data_cfg = yaml.safe_load(open(args.data_config_path, 'r'))
    train_data_loaders_dic, eval_data_loaders_dic, args.total_batch_size = get_dataset_loader_dict_new2(
        data_cfg)

    args.data_rep = data_cfg.get('data_rep', 'rot6d')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')

    # logging directory
    if not args.db:
        args.save_dir = create_latest_child_dir(args.save_dir)
        print(f"Saving to {args.save_dir}")
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # logging dashboard
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    dist_util.setup_dist(args.device)

    # model: load args.pretrained_model_path if it was supplied
    if args.pretrained_model_path != "":
        print(f"Loading pretrained model [{args.pretrained_model_path}]")
        path_model_args = os.path.join(
            os.path.dirname(args.pretrained_model_path), "args.json")
        if not os.path.exists(path_model_args):
            raise ValueError(f"Model path [{args.pretrained_model_path}] must be in the same" \
                            "directory as its model args file: [args.json]")
        with open(path_model_args, 'r') as f:
            args_pretrained_model = argparse.Namespace(**json.load(f))

        # check that the data_rep specifi
        assert args.data_rep==args_pretrained_model.data_rep, \
            f"pretrained model was trained on a different `data_rep` to what "\
            f"is supplied in [{args.data_config_path}]"

        # we will make `args_pretrained_model` the global args. But we want to
        # keep some config from `args` the same, so copy those values from
        # `args` to `args_pretrained_model`
        for k in [
                'save_dir', 'eval_hmr_every', 'update_hmr_every',
                'train_platform_type', 'log_interval', 'save_interval',
                'data_config_path', 'pretrained_model_path',
                'feature_mask_ratio', 'feature_mask_block_size',
                'feature_mask_training_exp', 'total_batch_size', 'num_iters_per_epoch'
        ]:
            setattr(args_pretrained_model, k, getattr(args, k))

        # now make the pretrained config the global config
        args = args_pretrained_model

        # load the pretrained model
        model, diffusion = create_emp_model_and_diffusion(
            args_pretrained_model, None)
        state_dict = torch.load(args.pretrained_model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

    # model: create new model of no args.model_path
    else:
        print("Creating new (untrained) model and diffusion...")
        model, diffusion = create_emp_model_and_diffusion(args, None)
        pretrained = True

    # print model properties to be easily read in slurm logs
    print("Model properties:")
    for k in [
            'data_rep',
            'arch',
            'feature_mask_ratio',
            'feature_mask_block_size',
            'cond_mode',
            'video_cond_input',
            'video_arch',
    ]:
        print(f"    {k}: [{getattr(args, k)}]")

    # some model logistics
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()
    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))

    # train
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, train_data_loaders_dic,
              eval_data_loaders_dic).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
