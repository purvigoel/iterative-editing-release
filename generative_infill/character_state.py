import numpy as np
import torch
import generative_infill.conversions as conversions
from generative_infill.summary_statistics import *
from mdm.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_euler_angles
import math

class CharacterState:
    def __init__(self, sample, model):
        self.sample = sample
        self.model = model

        self.joint_limits = { 
"right_knee": {"extend": {"index": 2, "operation": "-", "angle": {"neutral": 0, "full_extension": 20} }, "flex": {"index": 2, "operation": "-", "angle": {"neutral": 0, "full_extension": 150} } },
"left_knee": {"extend": {"index": 2, "operation": "-", "angle": {"neutral": 0, "full_extension": 20} }, "flex": {"index": 2, "operation": "-", "angle": {"neutral": 0, "full_extension": 150} } },
"right_hip": {"extend": {"index": 2, "operation": "-", "angle": {"neutral":7.92, "full_extension": 20} }, "flex": {"index": 2, "operation": "+", "angle": {"neutral": 7.92, "full_extension": -150} },
"abduct": {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension": 45} }, "adduct" : {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension": 20} }
},
"left_hip": {"extend": {"index": 2, "operation": "-", "angle": {"neutral": 7.92, "full_extension": 20} }, "flex": {"index": 2, "operation": "+", "angle": {"neutral": 7.92, "full_extension": -150} }, "abduct": {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension": 45} }, "adduct" : {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension": 20} }
},
"left_shoulder": {"adduct": {"index": 2, "operation": "-", "angle": {"neutral": -90, "full_extension": -30} }, "abduct": {"index": 2, "operation": "+", "angle": {"neutral": -90, "full_extension": 120} }, "flex": {"index": 0, "operation": "+", "angle": {"neutral": 0, "full_extension":90 }}, "extend": {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension":60 }}   },
"right_shoulder": {"adduct": {"index": 2, "operation": "-", "angle": {"neutral": -90, "full_extension": -30} }, "abduct": {"index": 2, "operation": "+", "angle": {"neutral": -90, "full_extension": 120}} , "flex": {"index": 0, "operation": "+", "angle": {"neutral": 0, "full_extension":90 }}, "extend": {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension":60 }}

},
"right_elbow": {"extend": {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension": -80} }, "flex": {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension": 20} } },
"left_elbow": {"extend": {"index": 0,"operation": "-", "angle": {"neutral": 0, "full_extension": 0} }, "flex": {"index": 0, "operation": "-", "angle": {"neutral": 0, "full_extension": 180} } },
"waist": {"extend": {"index": 2, "operation": "-", "angle": {"neutral":0, "full_extension": -30} }, "flex": {"index":2, "operation": "-", "angle": {"neutral": 0, "full_extension": -110}}},

}

    def get_pos(self):
        return conversions.get_pos_from_rep(self.model, self.sample)

    def get_important_frames(self, params, prev_frame_global,frange=[]):
        frames_to_edit = []
        for param in params:

            if param["type"] in ["rotation", "translation"]:
                frames_to_edit.append(param["frame"])
            elif param["type"] == "fix":
                frames_to_edit.append(param["start_time"])
                frames_to_edit.append(param["end_time"])
                #for f in range(param["start_time"] + 1, param["end_time"]):
                #    frames_to_edit.append(f)

        from_model_pos = self.get_pos()
        print("calculating important frames")
        extrema, _ = calc_summary_statistics(from_model_pos[0] , include_waist=False, time_range=[0,from_model_pos.shape[1]])
        extrema = extrema.tolist()
        print(extrema)        
        prev_frame = prev_frame_global.copy()
        for i in range(len(extrema)):
            potential_extrema = extrema[i]
            skip = False
            for curr_frame in frames_to_edit:
                if abs(curr_frame - potential_extrema) <= 4 :
                    skip = True
                    break
            if skip:
                continue
            already_covered = False
            print(prev_frame)
            for j in prev_frame:
                if abs(j - potential_extrema) <= 4:
                    already_covered = True
                    break
            if already_covered:
                continue
            prev_frame.append(potential_extrema)

        if len(frange) > 0:
            frange = np.array(frange)
            min_fr = np.min(frange)
            max_fr = np.max(frange)
            print("min/max", min_fr, max_fr)
            prev_frame_new = []
            for i in prev_frame:
                if i < max_fr + 15 and min_fr - 15 <= i:
                    prev_frame_new.append(i)
            prev_frame = prev_frame_new
        print(prev_frame)
        return prev_frame

    def step_character(self, type, method, sample, name, prev_edit, magnitude, frame, prev_frame_global = None, prev_data= None, sample_id=18, metadata=None, extrema_frames = []):
        prev_frame = prev_frame_global.copy()

        for i in range(len(extrema_frames)):
            potential_extrema = extrema_frames[i]
            if abs(frame - potential_extrema) <= 4 :
                continue
            already_covered = False
            for j in prev_frame:
                if abs(j - potential_extrema) <= 4:
                    already_covered = True
                    break
            if already_covered:
                continue
            prev_frame.append(potential_extrema)
        
        prev_frame.extend(required) 
        print("SAVING EXTREMA", prev_frame, extrema_frames, frame)
        if type == "rotate":
            sample = llm_method_to_program_local(self, method, sample, name, prev_edit, magnitude, frame, prev_frame, prev_data, sample_id)
        elif type == "translate":
            sample = llm_method_to_program_translate(self, method, sample, name, prev_edit, magnitude, frame, prev_frame, prev_data, sample_id, metadata)

        self.sample = sample.clone()
        return sample.clone()

    def rot6d_to_euler(self, rot6d):
        joint_rot6d = rotation_6d_to_matrix(rot6d.unsqueeze(0))
        joint_euler = matrix_to_euler_angles(joint_rot6d, "ZYX") * 180 / math.pi
        return joint_euler
    
    def get_joint_halfwid(self, joint, direction, frame, sample_id=0):
        frame_info = self.sample[sample_id, ..., frame].clone()
        right_hip = frame_info[2 * 6: 2 * 6 + 6, 0]
        left_hip = frame_info[1 * 6: 1 * 6 + 6, 0]
        right_knee = frame_info[5 * 6: 5 * 6 + 6, 0]
        left_knee = frame_info[6 * 6: 6 * 6 + 6, 0]
        com = frame_info[24*6:24*6+6, 0]
        right_shoulder = frame_info[17*6 : 17*6 + 6,0]
        left_shoulder = frame_info[16*6:16*6 + 6,0 ]
        waist = frame_info[0 : 6, 0]

        right_knee = frame_info[5*6 :5*6 + 6,0]
        left_knee = frame_info[4*6:4*6 + 6,0 ]

        right_elbow = frame_info[19*6:19*6 + 6,0 ]
        left_elbow = frame_info[18*6:18*6 + 6,0 ]

        right_hip_euler = self.rot6d_to_euler(right_hip)
        left_hip_euler = self.rot6d_to_euler(left_hip)
        left_shoulder_euler = self.rot6d_to_euler(left_shoulder)
        right_shoulder_euler = self.rot6d_to_euler(right_shoulder)
        left_knee_euler = self.rot6d_to_euler(left_knee)
        right_knee_euler = self.rot6d_to_euler(right_knee)
        right_elbow_euler = self.rot6d_to_euler(right_elbow)
        left_elbow_euler = self.rot6d_to_euler(left_elbow)
        waist_euler = self.rot6d_to_euler(waist)
        
        measurements = {
            "right_hip": { "mag": right_hip_euler, "min_x": "neutral, in a standing position."},
            "left_hip":  { "mag": left_hip_euler, "min_x": "neutral, in a standing position."},
            "right_shoulder": { "mag": right_shoulder_euler, "min_x": " outstretched at 90 degrees from the body.."},
            "left_shoulder":  { "mag": left_shoulder_euler, "min_x": " outstretched at 90 degrees from the body."},
            "right_knee": { "mag": right_knee_euler, "min_x": " outstretched at 90 degrees from the body.."},
            "left_knee":  { "mag": left_knee_euler, "min_x": " outstretched at 90 degrees from the body."},
            "right_elbow": { "mag": right_elbow_euler, "min_x": " outstretched at 90 degrees from the body.."},
            "left_elbow":  { "mag": left_elbow_euler, "min_x": " outstretched at 90 degrees from the body."},
            "waist": {"mag": waist_euler, "min_x": "what"},
        }
        desired_angle = 45
        if joint in self.joint_limits:
            if direction not in self.joint_limits[joint]:
                assert(1 == 0)
                return desired_angle
            measurement = measurements[joint]
            measurement_angle = measurement["mag"][0, self.joint_limits[joint][direction]["index"] ]
            measurement_operation = self.joint_limits[joint][direction]["operation"]
            limit_angle = self.joint_limits[joint][direction]["angle"]["full_extension"]
            
            if joint == "right_shoulder" and (direction == "flex" or direction == "extend"):
                measurement_angle *= -1
            
            ratio = 0.5
            desired_angle = (limit_angle * ratio + measurement_angle * (1.0 - ratio)) 
            desired_angle = desired_angle - measurement_angle
            
            if (joint == "left_elbow" or joint == "right_elbow") and direction == "extend":
                desired_angle *= -1

            if desired_angle < -90:
                desired_angle = -90
        return desired_angle

    def is_float(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def parse_mag(self, mag):
        words = mag.split(" ")
        for word in words:
            if self.is_float(word):
                return float(word)
        return 90
