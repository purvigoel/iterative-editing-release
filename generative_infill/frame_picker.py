import pose_mask
import numpy as np
from generative_infill.summary_statistics import *
from meo_constants import meoConstant

# the idea here is that if the edit describes an action, we need to edit some frame in the future. So we find
# the next extrema of the moving joint and edit that frame instead.
def handle_verb(sample_pos, frame, is_verb, joint, time_range=(0, meoConstant.max_frames), early_out=False):
    if is_verb:
        '''
        if "contact" in metadata:
            contact = metadata["contact"]
            query_joint = contact[1]
            metadata["query_joint"] = query_joint
        '''
        print(frame)
        if frame + 4 >= sample_pos.shape[1]:
            return (sample_pos.shape[1] - 1)
        #30
        _, extrema = calc_summary_statistics(sample_pos[0], ( min(frame + 5, time_range[1]), min(frame + 30, time_range[1]) ), frame, extra_joints = [joint])
        query_joint = joint
        frame_options = extrema[ query_joint ]
        
        best_q = -1
        frame = min(frame + 10, time_range[1])
        options = []
        shortcut_options = []
        if "hi_y_q" in frame_options:
            if best_q < frame_options["hi_y_q"]:
                best_q = frame_options["hi_y_q"]
                frame = frame_options["hi_y"]
            options.append(frame_options["hi_y"])
            if frame_options["hi_y_q"] > 0.1:
                shortcut_options.append(frame_options["hi_y"])
        if "lo_y_q" in frame_options:
            if best_q < frame_options["lo_y_q"]:
                best_q = frame_options["lo_y_q"]
                frame = frame_options["lo_y"]
            options.append(frame_options["lo_y"])
            if frame_options["lo_y_q"] > 0.1:
                shortcut_options.append(frame_options["lo_y"])

        if "lo_x_q" in frame_options:
            if best_q < frame_options["lo_x_q"]:
                best_q = frame_options["lo_x_q"]
                frame = frame_options["lo_x"]
            options.append(frame_options["lo_x"])
            if frame_options["lo_x_q"] > 0.1:
                shortcut_options.append(frame_options["lo_x"])

        if "hi_x_q" in frame_options:
            if best_q < frame_options["hi_x_q"]:
                best_q = frame_options["hi_x_q"]
                frame = frame_options["hi_x"]
            options.append(frame_options["hi_x"])
            if frame_options["lo_x_q"] > 0.1:
                shortcut_options.append(frame_options["hi_x"])
        if early_out:
            if len(shortcut_options) > 0:
                return shortcut_options
            return options

        if len(shortcut_options) > 0:
            frame = np.min(np.array(shortcut_options))
        print(frame, query_joint, frame_options)
        print("Found VERB, frame now", query_joint, frame)
    return frame

def get_current_frame():
    return globals()["curr_frame_interface"]

# when_joint, as_joint, before_joint, at_global_moment, at_frame
def at_frame(frame, joint, is_verb=False, motion=None):
    frame = handle_verb(motion, int(frame), is_verb, joint)
    return frame

def at_global_moment(param, joint, is_verb=False, motion=None):
    N = motion.shape[1]
    if param == "start_of_motion":
        frame = handle_verb(motion, 0, is_verb, joint)
        return frame
    elif param == "end_of_motion":
        frame = N - 1
        return frame
    elif param == "middle_of_motion":
        frame = int(N // 2)
        frame = handle_verb(motion, frame, is_verb, joint)
        return frame
    elif param == "entire_motion":
        return "entire_motion"
    else:
        return 30

def when_joint(joint, extrema, is_verb=None, motion=None, time_range=(0, meoConstant.max_frames)):
    if extrema == "highest" or extrema == "up":
        frame = int(time_range[0] + np.argmax(motion[0, time_range[0]:time_range[1], pose_mask.pose_dict[joint], 1]))
        return frame
    elif extrema == "lowest" or extrema == "down":
        frame = int(time_range[0] + np.argmin(motion[0, time_range[0]:time_range[1], pose_mask.pose_dict[joint], 1]))
        return frame
    elif extrema == "furthest_from_body":
        dist = motion[0, time_range[0]:time_range[1], pose_mask.pose_dict[joint],:] - motion[0,time_range[0]:time_range[1], 0,:]
        max_ind = -1
        max_val = -1
        dist[:, 1] = 0
        for i in range(time_range[0], time_range[1]):
            dist_from_body =np.linalg.norm(dist[i - time_range[0]])
            if dist_from_body > max_val:
                max_val = dist_from_body
                max_ind = i
        frame = max_ind
        return frame
    elif extrema == "closest_to_body":
        dist = motion[0,time_range[0]:time_range[1], pose_mask.pose_dict[joint],:] - motion[0,time_range[0]:time_range[1], 0,:]
        min_ind = -1
        min_val = 100000
        dist[:, 1] = 0
        for i in range(time_range[0], time_range[1]):
            dist_from_body = np.linalg.norm(dist[i - time_range[0]])
            if dist_from_body < min_val:
                min_val = dist_from_body
                min_ind = i
        frame = min_ind
        return frame

def as_joint(joint, extrema, is_verb=None, motion=None):
    # joint, extrema, is_verb=None, motion=None, time_range=(0, N)
    extrema_frame = when_joint(joint, extrema, is_verb=is_verb, motion = motion)

    return extrema_frame

def before_joint(joint, extrema, is_verb=None, motion=None):
    if not is_verb:
        frame = as_joint(joint, extrema, is_verb, motion=motion)
    else:
        # previous extrema 
        previous_frame = as_joint(joint, extrema, is_verb=False, motion=motion)
        
        # move back more
        frame = when_joint(joint, extrema, is_verb, motion=motion, time_range=(0, previous_frame))
    
    return frame

def after_joint(joint, extrema, is_verb=None, motion=None):
    # frame of the extrema
    frame = as_joint(joint, extrema, False, motion=motion)
    # extrema following.
    next_extrema_frame_options = handle_verb(motion, frame, is_verb=True, joint=joint, time_range=(0, meoConstant.max_frames),early_out=True)
    next_extrema_frame_options.sort()
    if len(next_extrema_frame_options) > 1:
       for i in range( 1, len(next_extrema_frame_options)):
            if next_extrema_frame_options[i] > frame:
                return next_extrema_frame_options[i]
    return next_extrema_frame_options[0]

