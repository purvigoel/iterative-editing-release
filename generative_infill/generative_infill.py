from jackson_import import *
from mdm.utils.fixseed import fixseed
import os
import argparse
import json
import numpy as np
import torch
from gthmr.emp_train.training_loop import TrainLoop
from mdm.utils.parser_util import train_ik_args, generate_args, train_args, train_emp_args
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from gthmr.emp_train.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
from mdm.utils.model_util import create_emp_model_and_diffusion
import shutil
from mdm.data_loaders.tensors import collate
from gthmr.lib.utils.mdm_utils import viz_motions
from VIBE.lib.dataset.vibe_dataset import rotate_about_D
from gthmr.emp_train.get_data import get_dataset_loader_dict_new2
from gthmr.lib.utils import data_utils
import ipdb
import yaml
from utils.misc import updata_ns_by_missing_keys
from mdm.utils.parser_util import generate_args, edit_args
from mdm.data_loaders import humanml_utils
from mdm.model.rotation2xyz import Rotation2xyz
import pose_mask
import matplotlib.pyplot as plt
from VIBE.lib.utils.eval_utils import compute_errors, compute_mpjpe, compute_g_mpjpe
import pickle
import math
import time
import os
from VIBE.lib.utils.geometry import rot6d_to_rotmat
from VIBE.lib.utils.geometry import rot6d_to_rotmat, batch_rodrigues
import mdm.utils.rotation_conversions as geometry

import generative_infill.model_loader as loader
from generative_infill.TrajectoryWrapper import *
from generative_infill.keyframer import *
#from generative_infill.InfillWrapper import *
from generative_infill.InfillWrapper_dpm import *

from generative_infill.SpatialTranslationWrapper import *
import generative_infill.io as io
from generative_infill.character_state import *
from generative_infill.reloader import *
from generative_infill.interpreter import *
import generative_infill.conversions as conversions
from generative_infill.save_stack import *
import openai_wrapper as llm
import generative_infill.to_meshes as mesh_converter
import requests
import logger as logger
from generative_infill.TimeWarper import *
import generative_infill.asset_library as asset_library
import time
from meo_constants import meoConstant

args = loader.load_args(1007)
device = 'cuda:1'
args.dataset="amass"
max_frames = meoConstant.max_frames
fps = 20
num_samples = 1
dist_util.setup_dist(args.device)
save_dir = "dump_results/"

os.makedirs(save_dir, exist_ok=True)

def load_all_models(trajectory=True, infill=True, spatial=True, uncond=True):

    data_template = loader.load_data(args, num_samples, max_frames)
    
    
    if trajectory:
        trajectory_fp = "mdm/meos/trajectory/model000585000.pt"
        trajectoryModel, trajectoryDiffusion = loader.load_model(trajectory_fp, device=dist_util.dev())
        trajectoryWrapper = TrajectoryWrapper(trajectoryModel, trajectoryDiffusion, data_template, dist_util, max_frames=max_frames)
    else:
        trajectoryWrapper = None

    if infill:
        infill_fp = "mdm/meos/infill/model000585000.pt"
        infillModel, infillDiffusion = loader.load_model(infill_fp, device=dist_util.dev())
        infillWrapper = InfillWrapper(infillModel, infillDiffusion, data_template, dist_util=dist_util, max_frames=max_frames)
    else:
        infillWrapper = None

    if uncond:
        uncond_fp = "mdm/meos/uncond/model000280000.pt"
        unconditionalMDMModel, unconditionalMDMDiffusion = loader.load_model(uncond_fp, device=dist_util.dev())
        unconditionalMDMWrapper = UnconditionalMDMWrapper(unconditionalMDMModel, unconditionalMDMDiffusion)
    else:
        unconditionalMDMWrapper = None

    if spatial:
        spatialEditHandler = SpatialEditHandler(unconditionalMDMWrapper,data_template,dist_util=dist_util, max_frames=max_frames)
    else:
        spatialEditHandler = None
	
    return trajectoryWrapper, infillWrapper, spatialEditHandler

def forward(sample, trajectoryWrapper, infillWrapper, spatialEditHandler, params, character_state, preserve_frame = False, postprocess=True):
    # spatial edit
    condition = sample.clone()

    edited_frames = []
    spatialEditHandler.prev_edit = prev_frame_global
    contains_root_edit = False
    need_infill = False
    need_copy = False
    copy_params = []
    print(params)
    for param in params:
        if param["type"] in ["rotation", "translation", "fix"]:
            sample, skip_infill = spatialEditHandler.forward(sample, param["type"], param["joint"], param["frame"], param, character_state, one_frame=False)
            if param["type"] == "fix":
                edited_frames.append(param["start_time"])
                if param["start_time"] != param["end_time"] and param["end_time"] == 60:
                    edited_frames.append(param["end_time"] - 1)
                
                for i in range(param["start_time"], param["end_time"] + 1): 
                    if i < max_frames:
                        edited_frames.append(i)
                copy_params.append(param)
            else:
                edited_frames.append(param["frame"])
            sample = reload(spatialEditHandler.translationWrapper.ikGuider.model, sample.clone())
            if param["joint"] == "waist":
                contains_root_edit = True
            
            if not skip_infill and len(edited_frames) < sample.shape[-1]:
                need_infill = True
    
    sample = reload(spatialEditHandler.translationWrapper.ikGuider.model, sample.clone())
    extrema = character_state.get_important_frames(params, prev_frame_global)
    print("Extrema:", extrema, "Edited frames:", edited_frames)
    if need_infill:
        trajectoryWrapper.preserve_frames = extrema
        trajectoryWrapper.preserve_key = condition.clone()
    
        infillWrapper.preserve_frames = extrema
        infillWrapper.preserve_key = condition.clone()

        keyframe_sample = sample.clone()

        window_size = 15
        trajectory_sample = trajectoryWrapper.forward(condition, sample, edited_frames, lo_window=window_size, hi_window=window_size)

        sample, condition = infillWrapper.forward(condition, keyframe_sample, trajectory_sample, window_size, window_size, edited_frames, is_root_edit=contains_root_edit)
        sample = reload(spatialEditHandler.translationWrapper.ikGuider.model, sample)

    for param in params:
        if param["type"] == "speed":
            timeWarper.preserve_frames = extrema
            sample = timeWarper.forward(sample, param["delta"], start_time=param["start"], end_time=param["end"])
            sample = torch.tensor(sample).to(condition.device)

    if postprocess:
        sample = spatialEditHandler.positionalConstrainer.ikSolver.fix_foot_penetration(sample, spatialEditHandler.translationWrapper.ikGuider.model, -1.08)

    sample =  reload(spatialEditHandler.translationWrapper.ikGuider.model, sample.clone())

    io.save_files(spatialEditHandler.translationWrapper.ikGuider.model, save_dir + "/", sample, condition, "synth_llm" + str(itr_id))

    if True: #preserve_frame:
        for param in params:
            if "frame" in param:
                prev_frame_global.append(param["frame"])
    return sample

def generative_infill(sample, params):
    trajectoryWrapper, infillWrapper, spatialEditHandler = load_all_models(trajectory=True, infill=True, spatial=True, uncond=True)
    character_state = CharacterState(sample, trajectoryWrapper.model)

    forward(sample, trajectoryWrapper, infillWrapper, spatialEditHandler, params=params, character_state=character_state)

def compiler_parse(compiler,saveStack, string):
    load_motion, save_motion = compiler.first_parse(string)
    sample = saveStack.load_motion(load_motion)

    motion_positions = conversions.get_pos_from_rep(spatialEditHandler.translationWrapper.ikGuider.model, sample)
    machine_code = compiler.parse(string, motion_positions)

    return sample, machine_code, save_motion

def set_up(sample):
    trajectoryWrapper, infillWrapper, spatialEditHandler = load_all_models(trajectory=True, infill=True, spatial=True, uncond=True)
    timeWarpWrapper = TimeWarpWrapper(spatialEditHandler.translationWrapper.ikGuider.model, max_frames)
    compiler = Compiler()
    return trajectoryWrapper, infillWrapper, spatialEditHandler, timeWarpWrapper, compiler

def set_up_character(sample):
    character_state = CharacterState(sample, trajectoryWrapper.model)
    saveStack = SaveStack()
    saveStack.save_motion("motion_0", sample)
    return character_state, saveStack

def send_get_request(url, params=None):
    response = requests.get(url, params=params)
    return response.text

def loop(execute, log_folder, render=True, postprocess=True):
    
    sample, params, out_name = compiler_parse(compiler, saveStack, execute)
    character_state = CharacterState(sample, trajectoryWrapper.model)
    
    start = time.time()
    rewrite_params = []
    need_clear = False
    for param in params:
        if param["type"] in ["rotation", "translation"] and param["frame"] == "entire_motion":
            extrema = character_state.get_important_frames([{"type":param["type"], "frame": -10}], prev_frame_global)
            already_edited = []
            for i in range(0, sample.shape[-1], 10):
                new_param = param.copy()
                new_param["frame"] = i
                rewrite_params.append(new_param)
                already_edited.append(i)
            new_param = param.copy()
            new_param["frame"] = sample.shape[-1] - 1
            rewrite_params.append(new_param)
            already_edited.append(sample.shape[-1] - 1)

            for extr in extrema:
                if extr in already_edited:
                    continue
                new_param = param.copy()
                new_param["frame"] = extr
                rewrite_params.append(new_param)
            need_clear = True
        else:
            rewrite_params.append(param)
    print(params)
    
    if need_clear:
        prev_frame_global.clear()
    
    if len(rewrite_params) > 0:
        sample = forward(sample, trajectoryWrapper, infillWrapper, spatialEditHandler, params=rewrite_params, character_state=character_state, postprocess=postprocess)
   
        saveStack.save_motion(out_name, sample)
        character_state = CharacterState(sample, trajectoryWrapper.model)
    print("Finished infill in", time.time() - start)
    #logger.save_motion(log_folder, str(itr_id), sample.cpu().numpy())
    #logger.save_motion(log_folder, str(itr_id) + "_positions", np.load(save_dir + "/synth_llm_iter_joints.npy"))
    #logger.save_motion(log_folder, str(itr_id) + "_condition", np.load(save_dir + "/synth_llm_iter_condition_joints.npy"))

    return sample

def llm_loop():
    llm.read_progprompt("")
    llm.get_incontext()
    while True:
        user_input = input("You: ")
        print("Chatbot:", user_input)
        prompt = user_input

        llm.prompt_sequence.append("# " + prompt + "\n")
        error_prompt_sequence = llm.prompt_sequence
        c, r = llm.query_model(llm.prompt_sequence, error_prompt_sequence, 0)
        llm.prompt_sequence.append(c)
        print(c)
        global itr_id
        loop(c, render=False, postprocess=False)
        itr_id +=1 

def llm_loop_initiate(sample_id, sample_is_motion=False):
    global saveStack
    global character_state
    if len(llm.prompt_sequence) == 0:
        llm.read_progprompt("")
        llm.get_incontext()
        print("initiating llm")
        if not sample_is_motion:
            sample = asset_library.asset_library[sample_id]
        else:
            sample = sample_id
        character_state, saveStack = set_up_character(sample)
    else:
        llm.prompt_sequence = []
        llm.read_progprompt("")
        llm.get_incontext()
        print("initiating llm")
        if not sample_is_motion:
            sample = asset_library.asset_library[sample_id]
        else:
            sample = sample_id
        character_state, saveStack = set_up_character(sample)


execute = '''
    '''
chatbot_text = False
chatbot_text_motion = True


prev_frame_global = []
itr_id = 0
saveStack = None
character_state = None
trajectoryWrapper, infillWrapper, spatialEditHandler, timeWarper, compiler = set_up(None)
logger = logger.Logger()


if chatbot_text_motion:
    source = asset_library.asset_library[2]
    llm_loop_initiate(source, sample_is_motion=True)
    logger.log_folder = "log_folder/"
    while True:
        user_input = input("You: ")
        print("Chatbot:", user_input)
        prompt = user_input
        
        llm.prompt_sequence.append("# " + prompt + "\n")
        error_prompt_sequence = llm.prompt_sequence
        c, r = llm.query_model(llm.prompt_sequence, error_prompt_sequence, 0)
        llm.prompt_sequence.append(c)
        print(c)
        loop(c, logger.log_folder, render=False, postprocess=False)
        itr_id +=1
elif chatbot_text:
    llm_loop_initiate(0, sample_is_motion=False)
    logger.log_folder = "log_folder/"
    while True:
        user_input = input("You: ")
        print("Chatbot:", user_input)
        prompt = user_input

        llm.prompt_sequence.append("# " + prompt + "\n")
        error_prompt_sequence = llm.prompt_sequence
        c, r = llm.query_model(llm.prompt_sequence, error_prompt_sequence, 0)
        llm.prompt_sequence.append(c)
        print(c)





