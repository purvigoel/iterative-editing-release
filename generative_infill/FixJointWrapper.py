import numpy as np
import generative_infill.ik_solver2 as IkSolver
import pose_mask
import torch
from mdm.utils.rotation_conversions import axis_angle_to_quaternion,axis_angle_to_6d,quaternion_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion, rotation_6d_to_aa
from VIBE.lib.utils.geometry import batch_rodrigues

class FixJointWrapper:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.model
        self.diffusion = model_wrapper.diffusion
        self.ikSolver = IkSolver.IKSolver()

    def fix_foot_slide(self, sample, start_frame, end_frame, side, force=False):
        if  start_frame == end_frame:
            return self.ikSolver.fix_foot_slide(sample, self.model, start_frame, side, force=force)
        else:
            return self.ikSolver.fix_foot_slide_range(sample, self.model, start_frame, end_frame, side, force=force)

    def ws_mag_to_local(self, sample, magnitude, frame):
        root_orient = sample[:, 0:6, ..., frame].clone().squeeze()
        root_orient_aa = rotation_6d_to_aa(root_orient).unsqueeze(0)
        root_orient_mat = batch_rodrigues(root_orient_aa).view( 3, 3)

        tmp_magnitude = magnitude.clone()
        tmp_magnitude[1] = 0.0
        if(magnitude[1] == 0.0):
            oldnorm = np.linalg.norm(tmp_magnitude.cpu().numpy())
            direc = tmp_magnitude.float() / oldnorm
            diff = torch.matmul(root_orient_mat.float(), direc)
            diff = diff * oldnorm
            tmp_magnitude = diff
        tmp_magnitude[1] += magnitude[1]
        return tmp_magnitude

    def closest_point_on_ray(self, ray_origin, ray_direction, point):
        # Normalize the ray direction
        ray_direction_normalized = ray_direction / np.linalg.norm(ray_direction)

        # Vector from the ray origin to the point
        vector_to_point = point - ray_origin
        # Project the vector_to_point onto the ray direction
        projection_length = np.dot(vector_to_point, ray_direction_normalized)
        projection_length = max(projection_length, 0)

        # Calculate the closest point
        closest_point = ray_origin + projection_length * ray_direction_normalized

        return closest_point

    # calculate procedural magnitudes
    def handle_mag(self, curr_joint_pos,  other_joint_pos, direction, sample, frame, joint):
        if(direction == "above"):
            above = np.array([0.0, 1.0, 0.0])
            above = torch.tensor(above).to(curr_joint_pos.device).cpu()
            loc = self.closest_point_on_ray( other_joint_pos.cpu(), above, curr_joint_pos.cpu())
            loc = loc.to(curr_joint_pos.device)
            print(loc - curr_joint_pos)
            return loc - curr_joint_pos
        elif (direction == "next_to"):
            if "right" in joint:
                next_to = np.array([-1.0, 0.0, 0.0])
            elif "left" in joint:
                next_to = np.array([1.0, 0.0, 0.0])
            next_to = torch.tensor(next_to).cpu().float() #.to(curr_joint_pos.device)
            next_to = self.ws_mag_to_local(sample.cpu(), next_to, frame)
            loc = self.closest_point_on_ray(other_joint_pos.cpu().float(), next_to, curr_joint_pos.cpu().float()).to(other_joint_pos.device)
            return loc - curr_joint_pos
        elif(direction == "away"):
            vec = curr_joint_pos - other_joint_pos
            distance = torch.norm(vec)
            vec = vec / distance
            loc = other_joint_pos + vec * (distance + 0.15)
            return loc - curr_joint_pos
        elif(direction == "towards"):
            vec = curr_joint_pos - other_joint_pos
            distance = torch.norm(vec)
            vec = vec / distance
            loc = other_joint_pos + vec * (distance - 0.15)
            return loc - curr_joint_pos
        elif direction == "in_front":
            above = np.array([0.0, 0.0, 1.0])
            above = torch.tensor(above).to(curr_joint_pos.device).cpu()
            above = self.ws_mag_to_local(sample.cpu(), above, frame)
            loc = self.closest_point_on_ray( other_joint_pos.cpu(), above, curr_joint_pos.cpu())
            loc = loc.to(curr_joint_pos.device)
            return loc - curr_joint_pos
        else:
            # contact constraint. positions here are calculated via joints, which are technically "under" the skin. just add a little offset here
            # so that the moving joint is above the skin
            mag = other_joint_pos - curr_joint_pos
            mag[1] += 0.001
            return mag

    # I'm sure there's a more extensible way to do this ;) If the waist changes position, we want to keep the feet fixed.
    def waist_to_target(self, sample, start_frame, end_frame, side, joint, other_joint, sample_pos, direction="contact"):
        edited_motion = sample.clone()
        if start_frame == end_frame:
            end_frame += 1
        for i in range(start_frame, end_frame + 1):
            if i >= sample_pos.shape[0]:
                break
            if other_joint != joint:
                other_joint_pos = sample_pos[i, pose_mask.pose_dict[other_joint], :]
                curr_joint_pos = sample_pos[i, pose_mask.pose_dict[joint], :]
            else:
                other_joint_pos = sample_pos[start_frame, pose_mask.pose_dict[other_joint], :]
                curr_joint_pos = sample_pos[i, pose_mask.pose_dict[joint], :]
            mag = self.handle_mag(curr_joint_pos, other_joint_pos, direction, sample, i, joint)
            edited_motion[:, 24*6:24*6 + 3, :, i] += mag.unsqueeze(0).unsqueeze(-1)

        
        sample_pos_edit = self.ikSolver.fk_shortcut(edited_motion, self.model)
        for i in range(start_frame, end_frame + 1):
            if i >= sample_pos.shape[0]:
                break

            mag = sample_pos[i, pose_mask.pose_dict["right_foot"], :] - sample_pos_edit[i, pose_mask.pose_dict["right_foot"], :]
            edited_motion = self.ikSolver.to_target(edited_motion, self.model, i, mag, "right_foot", ws=False, rewrite_target=True)
 
            
            mag = sample_pos[i, pose_mask.pose_dict["left_foot"], :] - sample_pos_edit[i, pose_mask.pose_dict["left_foot"], :]
            edited_motion = self.ikSolver.to_target(edited_motion, self.model, i, mag, "left_foot",ws=False, rewrite_target=True)
        
        return edited_motion

    def reposition_feet(self, old_motion, edited_motion, edit_frame):
        sample_pos = self.ikSolver.fk_shortcut(edited_motion, self.model)
        sample_pos_edit = self.ikSolver.fk_shortcut(old_motion, self.model)
        start_frame = edit_frame
        end_frame = edit_frame
        
        for i in range(start_frame, end_frame + 1):
            if i >= sample_pos.shape[0]:
                break

            mag = sample_pos[i, pose_mask.pose_dict["right_foot"], :] - sample_pos_edit[i, pose_mask.pose_dict["right_foot"], :]
            print(mag)
            if torch.abs(mag).sum() > 0.0:
                edited_motion = self.ikSolver.to_target(edited_motion, self.model, i, mag, "right_foot", ws=False, rewrite_target=True)
            mag = sample_pos[i, pose_mask.pose_dict["left_foot"], :] - sample_pos_edit[i, pose_mask.pose_dict["left_foot"], :]
            print(mag)
            if torch.abs(mag).sum() > 0.0:  
                edited_motion = self.ikSolver.to_target(edited_motion, self.model, i, mag, "left_foot",ws=False, rewrite_target=True)
        return edited_motion

    def ee_to_target(self, sample, start_frame, end_frame, side, joint, other_joint, sample_pos, direction="contact"):
        edited_motion = sample.clone()
        if start_frame == end_frame:
            end_frame += 1
        for i in range(start_frame, end_frame + 1):
            if i >= sample_pos.shape[0]:
                break
            if other_joint != joint:
                other_joint_pos = sample_pos[i, pose_mask.pose_dict[other_joint], :]
                curr_joint_pos = sample_pos[i, pose_mask.pose_dict[joint], :]
            else:
                other_joint_pos = sample_pos[start_frame, pose_mask.pose_dict[other_joint], :]
                curr_joint_pos = sample_pos[i, pose_mask.pose_dict[joint], :]
            mag = self.handle_mag(curr_joint_pos, other_joint_pos, direction, sample, i, joint)
            edited_motion = self.ikSolver.to_target(sample, self.model, i, mag, side)
        return edited_motion

    def forward(self, sample, param, fr=-1):
        side = param["joint"]
        print(param)
        skip_infill = False
        sample_pos = self.ikSolver.fk_shortcut(sample, self.model)

        # if the feet are getting fixed to the ground, this is like fixing foot slide. if waist is getting moved, keep the feet fixed. 
        if param["joint"] in ["right_foot", "left_foot"]:
            if param["other_joint"] == "ground":
                skip_infill = True
                sample = self.fix_foot_slide(sample, param["start_time"], param["end_time"], side, force=True)
            else:
                sample = self.ee_to_target(sample, param["start_time"], param["end_time"], side, param["joint"], param["other_joint"], sample_pos, param["direction"])

        elif param["joint"] in ["right_hand", "left_hand"]:
            sample = self.ee_to_target(sample, param["start_time"], param["end_time"], side, param["joint"], param["other_joint"], sample_pos, param["direction"])
        else:
            sample = self.waist_to_target(sample, param["start_time"], param["end_time"], side, param["joint"], param["other_joint"], sample_pos, param["direction"])

        return sample, skip_infill
