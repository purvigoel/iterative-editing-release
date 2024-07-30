from mdm.model.mdm import MDM
from gthmr.lib.models.emp import EMP
from mdm.diffusion import gaussian_diffusion as gd
from mdm.diffusion.respace import SpacedDiffusion, space_timesteps


def load_model_wo_clip(model, state_dict, BACK_COMPATIBLE=True):
    """
    Load the model. Handle some backcompatibility between models.
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                          strict=False)

    ## some backcompatibility stuff - probably remove this later ##
    if BACK_COMPATIBLE:
        # case where the model was trained before hmr_module was created
        if any([k.startswith('hmr_module.') for k in missing_keys]):
            print("*" * 80)
            print(
                "WARNING: HMR head not pretrained ... hmr eval metrics may be unreliable"
            )

        # case where the model was trained before EmbedVideo module was a different class
        if 'embed_video.weight' in unexpected_keys:

            def chang_key_name(state_dict, key_old, key_new):
                val = state_dict.pop(key_old)  # throws error if not present
                state_dict.update({key_new: val})

            chang_key_name(state_dict, 'embed_video.weight',
                           'embed_video.fc.weight')
            chang_key_name(state_dict, 'embed_video.bias',
                           'embed_video.fc.bias')
            missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                                  strict=False)

        # very similar to the prior case, except the video embedding linear layer name was changed
        if 'embed_video.fc.weight' in unexpected_keys:

            def chang_key_name(state_dict, key_old, key_new):
                val = state_dict.pop(key_old)  # throws error if not present
                state_dict.update({key_new: val})

            chang_key_name(state_dict, 'embed_video.fc.weight',
                           'embed_video.enc.weight')
            chang_key_name(state_dict, 'embed_video.fc.bias',
                           'embed_video.enc.bias')
            missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                                  strict=False)

    assert len(unexpected_keys) == 0
    print(missing_keys)
    assert all([(k.startswith('clip_model.') or k.startswith('hmr_module.'))
                for k in missing_keys])


def create_emp_model_and_diffusion(args, data):
    model = EMP(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    model.diffusion = diffusion
    return model, diffusion


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = args.cond_mode

    num_actions = 1

    if data is not None:
        if hasattr(data.dataset, 'num_actions'):
            num_actions = data.dataset.num_actions
    elif "action" in cond_mode:
        num_actions = 32 #16

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    # following is for datasets from the `vibe_datasets` class
    elif hasattr(args, "data_rep"):
        data_rep = args.data_rep
        if data_rep == 'rot6d': njoints, nfeats = 25, 6
        elif data_rep == 'rot6d_fc': njoints, nfeats = 154, 1
        elif data_rep == 'rot6d_fc_shape': njoints, nfeats = 164, 1
        elif data_rep == 'rot6d_fc_shape_axyz': njoints, nfeats = 236, 1
        elif data_rep == 'rot6d_fc_shape_axyz_avel': njoints, nfeats = 308, 1
        elif data_rep == 'rot6d_fc_shape_axyz_avel2': njoints, nfeats = 308, 1
        elif data_rep == 'rot6d_ks': njoints, nfeats = 524, 1
        elif data_rep == 'rot6d_ks_1': njoints, nfeats = 380, 1
        elif data_rep == 'rot6d_ks_2': njoints, nfeats = 452, 1
        else:
            raise
    if hasattr(args, "ifeats"):
        ifeats = args.ifeats
    else:
        ifeats = 236

    return {
        'modeltype': '',
        'njoints': njoints,
        'nfeats': nfeats,
        'num_actions': num_actions,
        'translation': True,
        'pose_rep': data_rep,
        'glob': True,
        'glob_rot': True,
        'latent_dim': args.latent_dim,
        'ff_size': 1024,
        'num_layers': args.layers,
        'num_heads': 4,
        'dropout': 0.1,
        'activation': "gelu",
        'data_rep': data_rep,
        'cond_mode': cond_mode,
        'cond_mask_prob': args.cond_mask_prob,
        'action_emb': action_emb,
        'arch': args.arch,
        'video_cond_input': args.video_cond_input,
        'video_arch': args.video_arch,
        'emb_trans_dec': args.emb_trans_dec,
        'clip_version': clip_version,
        'dataset': args.dataset,
        'feature_mask_ratio': args.feature_mask_ratio,
        'feature_mask_block_size': args.feature_mask_block_size,
        'feature_mask_training_exp': getattr(args, 'feature_mask_training_exp',
                                             0),
        'video_arch_experiment': getattr(args, 'video_arch_experiment', 0),
        'diffusion_steps': args.diffusion_steps,
        "ifeats": ifeats
    }


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON
                         if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=((gd.ModelVarType.FIXED_LARGE if not args.sigma_small
                         else gd.ModelVarType.FIXED_SMALL)
                        if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )
