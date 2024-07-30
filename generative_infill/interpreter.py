import numpy as np
import generative_infill.frame_picker as frame_picker
from meo_constants import meoConstant

cache = {}
def load_motion(name):
    return name

def save_motion(name):
    return name

def when_joint(joint, extrema, is_verb):
    return frame_picker.when_joint(joint, extrema, is_verb, motion_positions)

def after_joint(joint, extrema, is_verb):
    fr= frame_picker.after_joint(joint, extrema, is_verb, motion_positions)
    return fr

def get_current_frame():
    return meoConstant.curr_frame

def at_frame(frame, is_verb):
    return frame_picker.at_frame(frame, joint_global, is_verb, motion_positions)

def at_global_moment(name, is_verb):
    return frame_picker.at_global_moment(name, joint_global, is_verb, motion_positions)

def as_joint(joint, extrema, is_verb):
    return frame_picker.as_joint(joint, extrema, False, motion_positions)

def before_joint(joint, extrema, is_verb):
    return frame_picker.before_joint(joint, extrema, is_verb, motion_positions)

def do_rotate(joint, direction, time):
    return {"type":"rotation", "joint":joint, "frame":time, "direction":direction}

def do_translate(joint, direction, time):
    return {"type":"translation", "joint":joint, "frame":time, "direction":direction}

def do_relative_translate(joint, jointB, direction, start_time, end_time):
    if start_time == end_time and start_time != "entire_motion" and jointB != "ground":
        return {"type": "translation", "joint": joint, "frame": start_time, "direction": direction, "contact": [joint, jointB]}
    elif start_time == "entire_motion":
        location = jointB
        return {"type": "fix", "joint":joint, "other_joint":location, "start_time":0, "end_time":meoConstant.max_frames, "frame": None, "direction":direction}
    else:
        location=jointB
        return {"type": "fix", "joint":joint, "other_joint":location, "start_time":start_time, "end_time":end_time, "frame": None, "direction":direction}

def do_change_speed(speed, start_time, end_time):
    print(start_time, end_time)
    if start_time > end_time:
        start_time = 0
    return {"type": "speed", "delta": speed, "start": start_time, "end": end_time}

def do_fix_joint(joint, location, start_time, end_time):
    if start_time > end_time:
        start_time = 0
    return {"type": "fix", "joint":joint, "other_joint":location, "start_time":start_time, "end_time":end_time, "frame": None}

class Compiler:
    def __init__(self):
        self.motion = None
        self.motion_positions = None
        self.available_methods = ["do_relative_translate(", "do_translate(", "do_rotate(", "before_joint(", "as_joint(", "at_global_moment(", "at_frame(", "when_joint(", "load_motion(", "save_motion(", "do_change_speed(", "do_fix_joint("]
        globals()["curr_frame_interface"] = 0

    def first_parse(self, string):
        lines = string.splitlines()
        load_name = None
        save_name = None
        for line in lines:
            if "load_motion" in line:
                load_name = eval(line)
            if "save_motion" in line:
                save_name = eval(line)
        return load_name, save_name

    def parse(self, string, motion_positions):
        globals()["motion_positions"] = motion_positions 
        lines = string.splitlines()
        return_commands = []
        for line in lines:
            flag = False
            for available_method in self.available_methods:
                if available_method in line and line.strip().startswith(available_method):
                    flag = True
                    break
            if flag:
                if "load_motion" in line:
                    pass
                elif "save_motion" in line:
                    pass
                elif line:
                    globals()["joint_global"] = line.split("(")[1].split(",")[0].strip('\"')
                    print(joint_global)
                    ret = eval(line.rstrip())
                    return_commands.append(ret)
        return return_commands
        
    

