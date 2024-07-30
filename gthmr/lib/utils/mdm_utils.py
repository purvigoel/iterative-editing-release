from argparse import ArgumentParser
import numpy as np
import argparse
import os
import json
import joblib
import ipdb

import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion


def get_args_from_model_path(model_path):
    args = joblib.load('gthmr/extras/mdm_generate_args.pkl')
    args_to_overwrite = joblib.load('gthmr/extras/args_to_overwrite.pkl')

    # load args from model
    args.model_path = model_path

    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args:  # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print(
                'Warning: was not able to load [{}], using default value [{}] instead.'
                .format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


"""
Viz
"""

def viz_motions(num_samples,
                num_repetitions,
                out_path,
                all_motions,
                all_lengths=None,
                all_text=None,
                dataset='humanact12',
                fps=20,
                v2=False,
                view_params={'elev':120, 'azim':-90},
                coloring_seq = None,
                vis_mode = "default",
                extra_joints = None
                ):
    """
    Input
        all_motions -- shape == (B, J, 3, T). This contains the J=24 SMPL joints in the XYZ representation.  B = num_samples * num_repetitions.
    """
    if type(all_motions) is not np.ndarray:
        all_motions = no.array(all_motions)
        
    if all_text is None:
        all_text = [''] * len(all_motions)

    if all_lengths is None:
        all_lengths = [all_motions.shape[-1]] * len(all_motions)

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(True)

    skeleton = paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 5
    # fps = 20

    for sample_i in range(num_samples):
        rep_files = []
        for rep_i in range(num_repetitions):
            caption = all_text[rep_i * num_samples + sample_i]
            length = all_lengths[rep_i * num_samples + sample_i]
            motion = all_motions[rep_i * num_samples + sample_i].transpose(
                2, 0, 1)[:length]
            save_file = sample_file_template.format(sample_i, rep_i)
            # print(
            #     sample_print_template.format(caption, sample_i, rep_i,
            #                                  save_file))
            animation_save_path = os.path.join(out_path, save_file)
            # plot_3d_motion_func = plot_3d_motion_v2 if 'v2'else plot_3d_motion
            plot_3d_motion_func = plot_3d_motion

            c_seq = None if coloring_seq is None else coloring_seq[rep_i * num_samples + sample_i]
            if extra_joints is not None:
                extra = extra_joints[rep_i * num_samples + sample_i]
            else:
                extra = None
            plot_3d_motion_func(animation_save_path,
                           skeleton,
                           motion,
                           dataset=dataset,
                           title=caption,
                           fps=fps,
                           view_params=view_params,
                           coloring_seq=c_seq,
                           vis_mode = vis_mode,
                           extra_joints = extra
                           )
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(
            num_samples, num_repetitions, out_path, row_print_template,
            all_print_template, row_file_template, all_file_template, caption,
            num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(num_samples, num_repetitions, out_path,
                          row_print_template, all_print_template,
                          row_file_template, all_file_template, caption,
                          num_samples_in_out_file, rep_files, sample_files,
                          sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={num_repetitions}' if num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
        ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    # print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i +
            1) % num_samples_in_out_file == 0 or sample_i + 1 == num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(
            sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        # print(
        #     all_print_template.format(sample_i - len(sample_files) + 1,
        #                               sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(
            sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template
