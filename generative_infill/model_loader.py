import numpy as np
import torch
from gthmr.emp_train.get_data import get_dataset_loader
from mdm.utils.parser_util import train_chatbot_args, train_ik_args, generate_args, train_args, train_emp_args
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils.fixseed import fixseed
import os
import argparse
import json
from mdm.utils.model_util import create_emp_model_and_diffusion

def load_args(seed):
    '''
    cmd4="python -m generative_infill.generative_infill \
                --data_config_path gthmr/emp_train/config/data/${data}.yml \
                --model_path $motion_model \
                --seed ${seed}
                --filename $indir
                --outname diffuseIK_itr${rollout}.npy
                --itr 2
                --blend 0.8
                --num_samples ${num_samples}
                --save_dir ${savedir}
    '''
    try:
        args = train_chatbot_args()
    except:
        args = train_ik_args()

    args.seed = seed
    args.save_dir = "test4/0_100_1/"
    args.num_samples=100
    args.data_config_path="gthmr/emp_train/config/data/amasshml_FcShapeAxyzAvel.yml"
    args.filename = "drop"
    args.rollout = 4
    fixseed(args.seed)
    return args

def load_model(model_path, num_samples=1, save_dir="test_refactor/", device="cpu", map_location=None):
    path_model_args = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.exists(path_model_args):
        raise ValueError(f"Model path [{model_path}] must be in the same" \
                        "directory as its model args file: [args.json]")
    with open(path_model_args, 'r') as f:
        args_pretrained_model = argparse.Namespace(**json.load(f))

    args_pretrained_model.total_batch_size = num_samples
    args_pretrained_model.save_dir = save_dir
    #args_pretrained_model = updata_ns_by_missing_keys(args_pretrained_model, args)

    model, diffusion = create_emp_model_and_diffusion(args_pretrained_model, None)
    model.to(device)
    model.rot2xyz.smpl_model.eval()
    print(f"Loading checkpoints from [{model_path}]...")
    if map_location:
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        state_dict = torch.load(model_path)
    load_model_wo_clip(model, state_dict)
    model.eval()
    return model, diffusion

def load_data(args, num_samples, max_frames):
    args.data_rep = "rot6d_fc_shape_axyz"
    num_samples = 10
    data = get_dataset_loader(name=args.dataset,
                                  batch_size=num_samples,
                                  num_frames=max_frames,
                                  data_rep = args.data_rep,
                                  split='db',
                                  hml_mode='train', # modes are ("gt","train","eval","text_only").
                                  )
    return data




