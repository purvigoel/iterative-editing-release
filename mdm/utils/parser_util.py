from argparse import ArgumentParser
import argparse
import os
import json
import ipdb

def parse_and_load_from_model(parser, model_path=None):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()

    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
        
    # load args from model
    if model_path is None and args.model_path is None:
        model_path = get_model_path_from_args()
    else:
        model_path = args.model_path

    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif args.cond_mode=='unconditional': # backward compitability
            args.unconstrained=True

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    
    if 'data_config_path' in model_args.keys():
      args.data_config_path = model_args['data_config_path']
      args.emp = True
    else:
      args.emp = False
    if args.cond_mask_prob == 0:
        args.guidance_param = 1

    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--no_motion_augmentation", action='store_true',  
                       help="Turn off data augmentation for humanml model")
    group.add_argument("--use_amass_cont6d", action='store_true',  
                       help="")

def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--cond_mode", default='video', choices=['motion',"motion_action", 'video','unconditional','text','action','keyframe'], type=str,
                       help="MDM conditioning mode. Default is video.")
    group.add_argument("--video_cond_input", default='concat', choices=['concat','add','none'], type=str,
                       help="MDM conditioning mode. Default is video.")
    group.add_argument("--video_arch", default='linear', choices=['linear','mlp','trans_enc'], type=str,
                       help="Architecture to transform the video encoder features.")
    group.add_argument("--feature_mask_ratio", default=0, type=float, help="Video embedding: ratio of sequence to mask. Default 0 is no masking")
    group.add_argument("--feature_mask_block_size", default=5, type=int, help="Video embedding masking size. Ignored if feature_mask_ratio=0")
    group.add_argument("--feature_mask_training_exp", default=0, type=float)
    group.add_argument("--video_arch_experiment", default=0, type=int, help="Int lookup for video architecture in experimentation")

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='', 
          # choices=['humanml', 'kit', 'humanact12', 'uestc','amass'], 
          type=str, help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--split", default='train', type=str)
    group.add_argument("--rotation_augmentation", action='store_true',
                       help="Randomly rotate the motinos about the y axis")
    # TMP?
    group.add_argument("--data_rep", default="rot6d", type=str, help='legacy option for `train_mdm.py`')

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=10_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--num_workers", default=0, type=int,
                       help="Num_workers param from torch.utils.data.DataLoader")
    group.add_argument("--pretrained_model_path", default="", type=str, 
                        help="Pretrained `model.pt` file.")
    


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=False, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--condition_split", default='test', type=str,
                       help="Which dataset split to take for conditinoing (test/action/video")
    group.add_argument("--condition_dataset", default='', type=str,
                       help="Dataset for inference (using the conditioning)")
    group.add_argument("--num_sampling_frames", default=-1, type=int,
                       help="Number of frames in sample generation. Default -1 means the model will switch to dataset-level defaults")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")



def add_emp_options(parser):
    group = parser.add_argument_group('emp')
    group.add_argument("--data_config_path", required=True, type=str)
    group.add_argument("--db", default=0, type=int)
    group.add_argument("--eval_hmr_every", default=5, type=int)
    group.add_argument("--model_path", default='', type=str)
    group.add_argument("--use_dpm_solver", default=1, type=int)
    group.add_argument("--update_hmr_every", default=1, type=int)
    group.add_argument("--e_3d_loss_weight", default=60, type=int)
    group.add_argument("--e_pose_loss_weight", default=60, type=int)
    group.add_argument("--e_shape_loss_weight", default=60, type=int)
    group.add_argument("--db_n_batches", default=-1, type=int, help='A debugging option for evaluation script.  Use only n evaluation batches.')
    group.add_argument("--ow_eval_batch_size", default=-1, type=int, help='Used by eval.py to overwrite the batch size.')
    group.add_argument("--baseline_name", default='', type=str, help='Used in `20230303_evaluate_baseline.py`.')
    group.add_argument("--num_iters_per_epoch", default=100, type=int)
    # analysis
    group.add_argument("--overwrite_dpm_steps", type=int, default=-1)
    group.add_argument("--overwrite_dpm_order", type=int, default=-1)
    group.add_argument("--eval_full", type=int, default=1, help="by default, yes")
    group.add_argument("--eval_shortcut", type=int, default=0)
    
def add_infill_args(parser):
    group = parser.add_argument_group('emp')
    group.add_argument("--excite", default=5.0, type=float)
    group.add_argument("--x_direction", default=0.0, type=float)
    group.add_argument("--y_direction", default=1.0, type=float)
    group.add_argument("--z_direction", default=0.0, type=float)

def add_ik_args(parser):
    group = parser.add_argument_group('emp')
    group.add_argument("--filename", type=str)
    group.add_argument("--xloc", default=-1000, type=float)
    group.add_argument("--yloc", default=-1000, type=float)
    group.add_argument("--zloc", default=-1000, type=float)
    group.add_argument("--outname", type=str)
    group.add_argument("--blend", default = 0.0, type=float)
    group.add_argument("--itr", default = 0, type=int)
    group.add_argument("--startname", type=str)
    group.add_argument("--limb", type=str)
    group.add_argument("--num_samples", default=100, type=int)
    group.add_argument("--framenum", default=1, type=int)

def train_emp_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_emp_options(parser)
    return parser.parse_args()

def train_infill_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_emp_options(parser)
    add_infill_args(parser)
    return parser.parse_args()

def train_ik_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_emp_options(parser)
    add_ik_args(parser)
    return parser.parse_args()

def train_chatbot_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    #add_training_options(parser)
    #add_emp_options(parser)
    add_ik_args(parser)
    return parser.parse_args()

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    args.condition_dataset = args.dataset \
        if args.condition_dataset=="" \
        else args.condition_dataset
    
    return args

def generate_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    return parser


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)
