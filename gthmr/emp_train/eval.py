"""
    python -m gthmr.emp_train.eval \
        --dataset h36m \
        --data_config_path gthmr/emp_train/config/data/h36m_only.yml \
        --save_dir mdm/save/tmp \
        --model_path mdm/save/20230227_trainemp_exp14-0-h36m_only/000000/model000090000.pt \
        --db 0 \
        --use_dpm_solver 1 
"""
import os
import json
import ipdb
import torch
import json
import copy
import argparse
from mdm.utils.fixseed import fixseed
from mdm.utils.parser_util import train_emp_args
from mdm.utils import dist_util
from gthmr.emp_train.training_loop import TrainLoop
from gthmr.emp_train.get_data import get_dataset_loader_dict_new2
from mdm.utils.model_util import create_emp_model_and_diffusion, load_model_wo_clip
from mdm.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from nemo.utils.exp_utils import create_latest_child_dir
import yaml
from utils.misc import updata_ns_by_missing_keys


def main():
    args = train_emp_args()
    fixseed(args.seed)

    device = 'cuda'

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and parse data config
    data_cfg = yaml.safe_load(open(args.data_config_path, 'r'))

    _, eval_data_loaders_dic, args.total_batch_size = get_dataset_loader_dict_new2(
        data_cfg, eval_only=True)
        
    path_model_args = os.path.join(os.path.dirname(args.model_path),
                                   "args.json")
    if not os.path.exists(path_model_args):
        raise ValueError(f"Model path [{args.model_path}] must be in the same" \
                        "directory as its model args file: [args.json]")
    with open(path_model_args, 'r') as f:
        args_pretrained_model = argparse.Namespace(**json.load(f))

    args_pretrained_model.total_batch_size = args.total_batch_size
    
    # Overwrite save_dir
    args_pretrained_model.save_dir = args.save_dir
        

    # Backward comp
    args_pretrained_model = updata_ns_by_missing_keys(args_pretrained_model, args)

    # Load Pretrained Model
    print("creating model and diffusion...")
    model, diffusion = create_emp_model_and_diffusion(args_pretrained_model, None)
    model.to(device)
    model.rot2xyz.smpl_model.eval()

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    
    # Get a trainer object
    train_platform = None
    train_data_loaders_dic = None
    trainer = TrainLoop(args_pretrained_model, train_platform, model, diffusion,
                        train_data_loaders_dic, eval_data_loaders_dic)

    
    trainer.model.eval()
    if args.eval_full:
        trainer.validate_and_evaluate_all(0)

    if args.eval_shortcut:
        trainer.eval_shortcut()

    # trainer.validate_hmr(eval_data_loaders_dic['hmr']['h36m_1'], n_batches=args.db_n_batches)
    # trainer.evaluate_hmr(0)



if __name__ == "__main__":
    main()
