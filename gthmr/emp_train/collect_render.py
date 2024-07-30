"""
    python -m gthmr.emp_train.render \
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
from jackson_import import *


def main():
    args = train_emp_args()
    fixseed(args.seed)

    device = 'cuda'

    # os.makedirs(args.save_dir, exist_ok=True)
    args.save_dir = create_latest_child_dir(args.save_dir)
    print("[EXP DIR]: ", args.save_dir)

    # Load and parse data config
    data_cfg = yaml.safe_load(open(args.data_config_path, 'r'))
    _, eval_data_loaders_dic, args.total_batch_size = get_dataset_loader_dict_new2(
        data_cfg, eval_only=True)
    args.data_rep = data_cfg.get('data_rep', 'rot6d')

    # Load a dummy model for FK
    print("creating model and diffusion...")
    model, diffusion = create_emp_model_and_diffusion(args, None)
    model.to(device)
    model.rot2xyz.smpl_model.eval()

    args_pretrained_model = args
    nemo = False
    if args.baseline_name == 'glamr':
        if 'nemomocap' in list(data_cfg['eval_sets']['hmr'][0].keys())[0]:
            GLAMR_RES_DIR = '/home/groups/syyeung/wangkua1/data/mymocap/GLAMR/'
            all_data = joblib.load(osp.join(GLAMR_RES_DIR, 'nemomocap_all.pkl'))
        nemo=True

    elif args.baseline_name == 'nemo':
        assert 'nemomocap' in list(data_cfg['eval_sets']['hmr'][0].keys())[0]
        NEMO_RES_ROOT = '/home/groups/syyeung/wangkua1/nemo/out/eval'
        all_data = joblib.load(osp.join(NEMO_RES_ROOT, 'nemomocap_all.pkl'))
        nemo = True

    elif args.baseline_name == 'vibe':
        if 'nemomocap' in list(data_cfg['eval_sets']['hmr'][0].keys())[0]:
            RES_DIR = '/home/groups/syyeung/wangkua1/nemo/exps/'
            all_data = joblib.load(osp.join(RES_DIR, 'nemomocap_vibe_all.pkl'))

    elif args.baseline_name == 'pare':
        assert 'nemomocap' in list(data_cfg['eval_sets']['hmr'][0].keys())[0]
        RES_DIR = '/home/groups/syyeung/wangkua1/nemo/exps/'
        all_data = joblib.load(osp.join(RES_DIR, 'nemomocap_pare_all.pkl'))

    else:  # Our model
        path_model_args = os.path.join(os.path.dirname(args.model_path),
                                       "args.json")
        if not os.path.exists(path_model_args):
            raise ValueError(f"Model path [{args.model_path}] must be in the same" \
                            "directory as its model args file: [args.json]")
        with open(path_model_args, 'r') as f:
            args_pretrained_model = argparse.Namespace(**json.load(f))

        args_pretrained_model.total_batch_size = args.total_batch_size
        args_pretrained_model.save_dir = args.save_dir

        # Load Pretrained Model
        print("creating model and diffusion...")
        model, diffusion = create_emp_model_and_diffusion(args_pretrained_model,
                                                          None)
        model.to(device)
        model.rot2xyz.smpl_model.eval()

        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

    # Get a trainer object
    train_platform = None
    train_data_loaders_dic = None
    trainer = TrainLoop(args_pretrained_model, train_platform, model,
                        diffusion, train_data_loaders_dic,
                        eval_data_loaders_dic)

    trainer.model.eval()
    eval_loader = trainer.eval_hmr_data_loader_dict['nemomocap_val']
    if args.baseline_name == '':
        trainer.validate_hmr(eval_loader)
    else:
        all_data['prefix'] = f'{args.baseline_name}'
        trainer.validate_baseline(all_data)

    evaluation_accumulators = trainer.evaluation_accumulators

    for k, v in evaluation_accumulators.items():
        if len(evaluation_accumulators[k]) > 0:
            evaluation_accumulators[k] = np.vstack(v)
    """
    ipdb> evaluation_accumulators.keys()
    dict_keys(['pred_j3d', 'pred_j3d_nobeta', 'target_j3d', 'target_j3d_mosh', 'target_j3d_mosh_nobeta', 'target_theta', 'pred_verts', 'pred_slv_j3d', 'target_slv_j3d', 'pred_g_j3d', 'target_g_j3d', 'pred_g_j3d_nobeta', 'target_g_j3d_mosh', 'target_g_j3d_mosh_nobeta', 'vid_name', 'idxs', 'motions', 'pred_motions', 'img_name', 'pred_g_v3d', 'pred_g_v3d_nobeta', 'target_g_v3d', 'orig_trans'])
    """
    # all_data = evaluation_accumulators
    # batch_idx = 0
    # frame_idx = 0
    # img_name = all_data['img_name'][batch_idx, frame_idx, 0]
    # verts = all_data['pred_g_v3d'][batch_idx, frame_idx]
    # orig_trans = all_data['orig_trans'][batch_idx, frame_idx]
    # verts += orig_trans

    
    joblib.dump(evaluation_accumulators, osp.join(args.save_dir, 'output.p'))

if __name__ == "__main__":
    main()
