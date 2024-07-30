import numpy as np
import torch
import pose_mask
from meo_constants import meoConstant

def calc_summary_statistics(motion, time_range=[0,meoConstant.max_frames], qframe=None, include_waist =True, extra_joints = []):
    summary = {}
    if include_waist:
        joints = ["right_foot", "left_foot", "right_hand", "left_hand", "waist"]
    else:
        joints = ["right_foot", "left_foot", "right_hand", "left_hand"]
    if len(extra_joints) > 0:
        joints.extend(extra_joints)
    important_frames = []
    if time_range[0] == (motion.shape[0] - 1):
        time_range = ( motion.shape[0] - 1, motion.shape)
    if time_range[0] == time_range[1]:
        time_range = (time_range[0] , time_range[0] + 1)
    print(time_range)
    for joint in joints:
        summary[joint] = {}
        frame = int(time_range[0] + np.argmin(motion[ time_range[0]:time_range[1], pose_mask.pose_dict[joint], 1]))
        lo_frame = frame
        min_val = np.min(motion[ time_range[0]:time_range[1], pose_mask.pose_dict[joint], 1])

        frame = int(time_range[0] + np.argmax(motion[ time_range[0]:time_range[1], pose_mask.pose_dict[joint], 1]))
        hi_frame = frame
        max_val = np.max(motion[ time_range[0]:time_range[1], pose_mask.pose_dict[joint], 1])

        frame = int(time_range[0] + np.argmax(motion[ time_range[0]:time_range[1], pose_mask.pose_dict[joint], 0]))
        far_frame = frame
        far_val = np.max(motion[ time_range[0]:time_range[1], pose_mask.pose_dict[joint], 0])

        frame = int(time_range[0] + np.argmin(motion[ time_range[0]:time_range[1], pose_mask.pose_dict[joint], 0]))
        close_frame = frame
        close_val = np.min(motion[ time_range[0]:time_range[1], pose_mask.pose_dict[joint], 0])

        if joint != "waist":
            root_y_min = motion[lo_frame, 0, 1]
            root_y_max = motion[hi_frame, 0, 1]
            root_x_min = motion[close_frame, 0, 0]
            root_x_max = motion[far_frame, 0, 0]

            max_val -= root_y_max
            min_val -= root_y_min
            far_val -= root_x_max
            close_val -= root_x_min

        y_sig = max_val - min_val
        x_sig = far_val - close_val
        print(joint, y_sig, x_sig, time_range)
        important = []
        if y_sig >= 0.4 or joint in extra_joints:
            important.append(lo_frame)
            important.append(hi_frame)
            summary[joint]["lo_y"] = lo_frame
            summary[joint]["hi_y"] = hi_frame
            summary[joint]["y_sig"] = y_sig
            if qframe:
                summary[joint]["lo_y_q"] = np.abs(min_val - motion[qframe, pose_mask.pose_dict[joint], 1])
                summary[joint]["hi_y_q"] = np.abs(max_val - motion[qframe, pose_mask.pose_dict[joint], 1])
        if x_sig >= 0.4 or joint in extra_joints:
            important.append(far_frame)
            important.append(close_frame)
            summary[joint]["lo_x"] = close_frame
            summary[joint]["hi_x"] = far_frame
            summary[joint]["x_sig"] = x_sig
            if qframe:
                summary[joint]["lo_x_q"] = np.abs(close_val - motion[qframe, pose_mask.pose_dict[joint], 0])
                summary[joint]["hi_x_q"] = np.abs(far_val - motion[qframe, pose_mask.pose_dict[joint], 0])
        important_frames.extend(important)
    
    important_frames = np.unique(np.array(important_frames))
    print(summary, important_frames)
    return important_frames, summary
