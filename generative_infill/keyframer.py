from generative_infill.SpatialTranslationWrapper import *
from generative_infill.rotations import *
import pose_mask

from VIBE.lib.utils.geometry import rot6d_to_rotmat
from VIBE.lib.utils.geometry import rot6d_to_rotmat, batch_rodrigues
import mdm.utils.rotation_conversions as geometry

from generative_infill.param_translator import *
from generative_infill.FixJointWrapper import *
from meo_constants import meoConstant
from mdm.utils.rotation_conversions import axis_angle_to_quaternion,axis_angle_to_6d,quaternion_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion, rotation_6d_to_aa

class SpatialEditHandler:
    def __init__(self, ikGuider, data_template, num_samples=1, dist_util=None, max_frames=meoConstant.max_frames):
        self.data_template = data_template
        self.text_place_holder = [''] * num_samples
        self.num_samples = num_samples
        self.max_frames = max_frames
        print(max_frames)
        self.prev_edit = []
        self.translationWrapper = SpatialTranslationWrapper(ikGuider, data_template, dist_util)
        self.positionalConstrainer = FixJointWrapper(ikGuider)

    def get_fudge_vector(self, direction):
        if direction == "next_to":
            return -0.1, 0.0, 0.0
        else:
            return 0.0, 0.05, 0.0

    def forward(self, sample, edit_type, edit_joint, edit_frame, params, character_state, one_frame=False):
        iterator = iter(self.data_template)
        input_motions, model_kwargs = next(iterator)
        
        model_kwargs['y']['text'] = self.text_place_holder

        input_motions = sample.float().clone()
        all_motions = input_motions.clone()
        keyframe = input_motions.clone()

        skip_infill = False
        print(edit_joint, edit_type)
        if edit_type == "rotation":
            degrees = character_state.get_joint_halfwid(edit_joint, params["direction"], params["frame"])
            params["degrees"] = degrees
            params = params_rotation_adjust(params)
            print("Setting degrees to ", degrees)
            edit_direction = params["direction"]
            sample = self.rotation_edit(input_motions, edit_joint,edit_direction, edit_frame, degrees)
            #if params["joint"] == "waist":
            #    sample = self.positionalConstrainer.reposition_feet(all_motions, sample, edit_frame)
            
            #character_state.sample = sample.clone()
        elif edit_joint == "waist" and edit_type != "fix":
            params = params_waist_adjust(params)
            edit_joint = "ground"
            degrees = params["degrees"]
            edit_direction = params["direction"]
            old_motions = input_motions.clone()
            sample = self.rotation_edit(input_motions, edit_joint,edit_direction, edit_frame, degrees)
        elif edit_type == "fix":
            # positional constraint
            if "waist" in params["joint"]:
                sample, skip_infill = self.positionalConstrainer.forward(input_motions, params)
            elif "foot" in params["joint"]:
                sample, skip_infill = self.positionalConstrainer.forward(input_motions, params)
            else:
                pos = self.positionalConstrainer.ikSolver.fk_shortcut(input_motions, self.positionalConstrainer.model)
                
                #magx, magy, magz = self.get_fudge_vector(params["direction"])
                contact = [params["joint"], params["other_joint"], -1]
                
                sample_trans = input_motions.clone()
                if one_frame:
                    end_time = params["start_time"]
                else:
                    end_time = params["end_time"]


                if params["joint"] == "right_hand":
                    angles = pose_mask.right_arm_angles
                else:
                    angles = pose_mask.left_arm_angles

                if params["joint"] in ["right_hand", "left_hand"]:
                    fr = params["start_time"]
                    joint = params["joint"]
                    other_joint = params["other_joint"]
                    for fr in range(params["start_time"], end_time + 1):
                        if fr >= pos.shape[0]:
                            break
                        joint = params["joint"]
                        other_joint = params["other_joint"]
                        if other_joint == joint:
                            target = pos[params["start_time"], pose_mask.pose_dict[other_joint], :]
                        elif other_joint == "ground":
                            target = pos[params["start_time"], pose_mask.pose_dict[joint], :]
                            target[1] = pos[params["start_time"], pose_mask.pose_dict["right_foot"], 1]
                        else:
                            target = pos[fr, pose_mask.pose_dict[other_joint], :]
                        print(target)                        
                        if "direction" not in params:
                            params["direction"] = None

                        mag = self.handle_mag(pos[fr, pose_mask.pose_dict[joint], :], target, params["direction"], sample_trans, fr, joint, other_joint)
                        magx = mag[0]
                        magy = mag[1]
                        magz = mag[2]
                        print("MAGS", magx, magy, magz)

                        use_diff_ik = False
                        if not use_diff_ik:
                            sample_trans, skip_infill = self.positionalConstrainer.forward(sample, params, fr)
                            break
                        else:
                            sample_trans = self.translation_edit(sample_trans, fr, joint, magx, magy, magz, contact=contact).to(sample.device)
                    
                sample = sample_trans
                
        else:
            assert edit_joint in ["right_foot", "left_foot", "right_hand", "left_hand", "waist"]
            params = params_translation_adjust(params)
            magx = params["magx"]
            magy = params["magy"]
            magz = params["magz"]
            contact = None
            if "contact" in params:
                contact = params["contact"]
            sample = self.translation_edit(input_motions, edit_frame, edit_joint, magx, magy, magz, contact)
       

        return sample, skip_infill

    def closest_point_on_ray(self, ray_origin, ray_direction, point):
        # Normalize the ray direction
        if torch.is_tensor(ray_direction):
            ray_direction = ray_direction.cpu().numpy()
        if torch.is_tensor(ray_origin):
            ray_origin = ray_origin.cpu().numpy()
        if torch.is_tensor(point):
            point = point.cpu().numpy()
        ray_direction_normalized = ray_direction / np.linalg.norm(ray_direction)

        # Vector from the ray origin to the point
        vector_to_point = point - ray_origin
        # Project the vector_to_point onto the ray direction
        projection_length = np.dot(vector_to_point, ray_direction_normalized)
        projection_length = max(projection_length, 0)

        # Calculate the closest point
        closest_point = ray_origin + projection_length * ray_direction_normalized

        return closest_point

    '''
    def handle_mag(self, curr_joint_pos,  other_joint_pos, direction):
        if(direction == "above"):
            above = np.array([0.0, 1.0, 0.0])
            above = torch.tensor(above).to(curr_joint_pos.device)
            loc = self.closest_point_on_ray( other_joint_pos, above, curr_joint_pos)
            return loc - curr_joint_pos
        elif (direction == "next_to"):
            next_to = np.array([-1.0, 0.0, 0.0])
            next_to = torch.tensor(next_to).cpu().float() #.to(curr_joint_pos.device)
            loc = self.closest_point_on_ray(other_joint_pos.cpu().float(), next_to, curr_joint_pos.cpu().float()).to(other_joint_pos.device)
            return loc - curr_joint_pos
        else:
            mag = other_joint_pos - curr_joint_pos
            mag[1] += 0.05
            return mag
    '''
    def ws_mag_to_local(self, sample, magnitude, frame, is_head=False):
        print(is_head)
        root_orient = sample[:, 0:6, ..., frame].clone().squeeze()
        root_orient_aa = rotation_6d_to_aa(root_orient).unsqueeze(0)
        root_orient_mat = batch_rodrigues(root_orient_aa).view( 3, 3)

        if is_head:
            head_orient = sample[:, 15*6 : 15*6 + 6, ..., frame].clone().squeeze()
            head_orient_aa = rotation_6d_to_aa(head_orient).unsqueeze(0)
            head_orient_mat = batch_rodrigues(head_orient_aa).view( 3, 3)

        tmp_magnitude = magnitude.clone()
        tmp_magnitude[1] = 0.0
        if(magnitude[1] == 0.0):
            oldnorm = np.linalg.norm(tmp_magnitude.cpu().numpy())
            direc = tmp_magnitude.float() / oldnorm

            if is_head:
                root_orient_mat = torch.matmul(root_orient_mat, head_orient_mat)

            diff = torch.matmul(root_orient_mat.float(), direc)
            diff = diff * oldnorm
            tmp_magnitude = diff
        tmp_magnitude[1] += magnitude[1]
        return tmp_magnitude

    def handle_mag(self, curr_joint_pos,  other_joint_pos, direction, sample, frame, joint, other_joint=None):
        if(direction == "above"):
            above = np.array([0.0, 1.0, 0.0])
            above = torch.tensor(above).to(curr_joint_pos.device)
            loc = self.closest_point_on_ray( other_joint_pos + above * 0.2, above, curr_joint_pos)
            loc = torch.tensor(loc).to(curr_joint_pos.device)
            return loc - curr_joint_pos
        elif (direction == "next_to"):
            if "right" in joint:
                next_to = np.array([-1.0, 0.0, 0.0])
            elif "left" in joint:
                next_to = np.array([1.0, 0.0, 0.0])
            elif "waist" in joint:
                next_to = np.array([0.0, 0.0, 1.0])
            next_to = torch.tensor(next_to).cpu().float()
            next_to = self.ws_mag_to_local(sample.cpu(), next_to, frame)
            loc = torch.tensor(self.closest_point_on_ray(other_joint_pos.cpu().float() + next_to * 0.2, next_to, curr_joint_pos.cpu().float())).to(other_joint_pos.device)
            return loc - curr_joint_pos
        elif direction == "in_front":
            next_to = np.array([0.0, 0.0, 1.0])
            next_to = torch.tensor(next_to).cpu().float()
            is_head = False
            if other_joint and other_joint == "head":
                is_head = True
            
            next_to = self.ws_mag_to_local(sample.cpu(), next_to, frame, is_head=is_head)
            loc = torch.tensor(self.closest_point_on_ray(other_joint_pos.cpu().float() + next_to * 0.25, next_to, curr_joint_pos.cpu().float())).to(other_joint_pos.device)
            return loc - curr_joint_pos
        else:
            direc = other_joint_pos - curr_joint_pos
            if direc.sum() > 0:
                dist = torch.norm(direc)
                direc = direc / torch.norm(direc)
                target = curr_joint_pos + 0.75 * dist * direc
                mag = target - curr_joint_pos
            else:
                mag = other_joint_pos - curr_joint_pos
            #mag[1] += 0.05
            return mag

    def rotation_edit(self, sample, edit_joint, edit_direction, edit_frame, degrees, i=0):
        input_motions = sample.clone()
        original = sample.clone()
        keyframe = sample.clone()
        frame = edit_frame
        direction = edit_direction
        if(edit_joint == "left_foot" or edit_joint == "left_hip"):
            tmatrix = transform_mat(direction, "foot","left", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)
            tmatrix = transform_mat(direction, "foot","left", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)
            keyframe[i] = transform(input_motions[i], tmatrix, [6, 7, 8, 9, 10, 11], keyframe[i].unsqueeze(0))[0]
        
        elif(edit_joint == "right_foot" or edit_joint == "right_hip"):
            tmatrix = transform_mat(direction, "foot", "right", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)
            keyframe[i] = transform(input_motions[i], tmatrix, [12, 13, 14, 15, 16, 17], keyframe[i].unsqueeze(0))[0]
        elif(edit_joint == "right_knee"):
            tmatrix = transform_mat(direction, "foot", "right", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)

            keyframe[i] = transform(input_motions[i], tmatrix, [30, 31, 32, 33, 34, 35], keyframe[i].unsqueeze(0))[0]
        
        elif(edit_joint == "left_knee"):
            tmatrix = transform_mat(direction, "foot", "left", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)
            keyframe[i] = transform(input_motions[i], tmatrix, [24, 25, 26, 27, 28, 29], keyframe[i].unsqueeze(0))[0]
        
        elif(edit_joint == "left_arm" or edit_joint == "left_shoulder"):
            tmatrix = transform_mat(direction, "hand", "left", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)
            keyframe[i] = transform(input_motions[i], tmatrix, [96, 97, 98, 99, 100, 101], keyframe[i].unsqueeze(0))[0]
        
        elif(edit_joint == "right_arm" or edit_joint == "right_shoulder"):
            tmatrix = transform_mat(direction, "hand", "right", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)

            keyframe[i] = transform(input_motions[i], tmatrix, [102, 103, 104, 105, 106, 107], keyframe[i].unsqueeze(0))[0]        


        elif(edit_joint == "left_collar"):
            tmatrix = transform_mat(direction, "hand", "left", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)
            keyframe[i] = transform(input_motions[i], tmatrix, [78, 79, 80, 81, 82, 83], keyframe[i].unsqueeze(0))[0]
            
        elif(edit_joint == "right_collar"):
            tmatrix = transform_mat(direction, "hand", "right", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)

            keyframe[i] = transform(input_motions[i], tmatrix, [84, 85, 86, 87, 88, 89], keyframe[i].unsqueeze(0))[0]
            
        elif(edit_joint == "left_elbow"):
            tmatrix = transform_mat(direction, "hand", "left", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)

            keyframe[i] = transform(input_motions[i], tmatrix, [108, 109, 110, 111, 112, 113], keyframe[i].unsqueeze(0))[0]
            
        elif(edit_joint == "right_elbow"):
            tmatrix = transform_mat(direction, "hand", "right", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)
        
            keyframe[i] = transform(input_motions[i], tmatrix, [114, 115, 116, 117, 118, 119], keyframe[i].unsqueeze(0))[0]
        
        elif(edit_joint == "waist"):
            tmatrix = transform_mat_waist(direction, "hand", "right", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)

            keyframe[i] = transform(input_motions[i], tmatrix, [0, 1, 2, 3, 4, 5], keyframe[i].unsqueeze(0))[0]

            left_leg_angles = input_motions[i, 6:12, :, edit_frame].squeeze().unsqueeze(0)
            left_leg_mat = geometry.rotation_6d_to_matrix(left_leg_angles)
            left_leg = torch.bmm(left_leg_mat, torch.transpose(tmatrix, 2, 1))

            left_leg_rot6d = matrix_to_rotation_6d(left_leg)
            keyframe[i, 6:12, ...] = left_leg_rot6d.unsqueeze(-1).unsqueeze(-1)

            right_leg_angles = input_motions[i, 12:18, :, edit_frame].squeeze().unsqueeze(0)
            right_leg_mat = geometry.rotation_6d_to_matrix(right_leg_angles)
            right_leg = torch.bmm(right_leg_mat, torch.transpose(tmatrix, 2, 1))

            right_leg_rot6d = matrix_to_rotation_6d(right_leg)
            keyframe[i, 12:18, ...] = right_leg_rot6d.unsqueeze(-1).unsqueeze(-1)

        elif(edit_joint == "ground"):
            if direction == "up":
                keyframe[i, 24* 6 + 1: 24 * 6 + 2, :, :] += degrees
            elif direction == "forward":
                keyframe[i, 24* 6 + 2: 24 * 6 + 3, :, :] += degrees
            elif direction == "backward":
                keyframe[i, 24* 6 + 2: 24 * 6 + 3, :, :] -= degrees
            elif direction == "out":
                keyframe[i, 24* 6 + 0: 24 * 6 + 1, :, :] += degrees
            T = keyframe.shape[-1]
            j_dic = self.translationWrapper.ikGuider.model.forward_kinematics(keyframe[i].unsqueeze(0), None)
            smpl_joints = j_dic['kp_45_joints'][:, :24].reshape(1, T, 24, 3 ) +  j_dic["pred_trans"].reshape(1, T,3).unsqueeze(-2)
            smpl_joints = smpl_joints +  torch.tensor([0, 0.2256, 0]).repeat(1, T, 1).unsqueeze(-2).to(smpl_joints.device)
            smpl_joints = smpl_joints.view(1, T, 24 * 3, 1)
            smpl_joints = smpl_joints.permute(0, 2, 3, 1)
            keyframe[i, 164 : 236, :, :] = smpl_joints.clone()               
            

        elif(edit_joint == "hips"):
            tmatrix = transform_mat(direction, "hips", "hips", sample.device, degrees)
            aa_transform = geometry.rotation_6d_to_aa(tmatrix)
            aa_transform = aa_transform.unsqueeze(0)
            tmatrix = batch_rodrigues(aa_transform).view(-1, 3, 3)

            keyframe[i] = transform(input_motions[i], tmatrix, [0, 1, 2, 3, 4, 5], keyframe[i].unsqueeze(0))[0]
        
        '''
        if edit_joint == "right_arm" or edit_joint == "right_elbow":
            sample = original
            sample[i, pose_mask.right_arm_angles, :, :] = pred[i, pose_mask.right_arm_angles, :, :]
        elif edit_joint == "left_arm" or edit_joint == "left_elbow":
            sample = original
            sample[i, pose_mask.left_arm_angles, :, :] = pred[i, pose_mask.left_arm_angles, :, :]
        elif edit_joint == "left_foot" or edit_joint == "left_knee":
            sample = original
            sample[i, pose_mask.left_leg_angles, :, :] = pred[i, pose_mask.left_leg_angles, :,:]
        elif edit_joint == "right_foot" or edit_joint == "right_ankle":
            sample = original
            sample[i, pose_mask.right_leg_angles, :, :] = pred[i, pose_mask.right_leg_angles, :, :]
        elif edit_joint == "right_knee":
            sample = original
            sample[i, pose_mask.right_leg_angles, :, :] = pred[i, pose_mask.right_leg_angles, :, :]
        elif edit_joint == "left_knee":
            sample = original
            sample[i, pose_mask.left_leg_angles, :, :] = pred[i, pose_mask.left_leg_angles, :, :]
        '''
        sample[i, ..., edit_frame]  = keyframe[i, ..., edit_frame]
        if edit_joint == "ground" or edit_joint == "hips":
            if edit_frame not in self.prev_edit:
                sample[i,  6:24*6, :, :] = 0
                sample[i, 25*6:, :, :] = 0
       
        
        return sample.clone()

    def translation_edit(self, motion, frame, edit_joint, magx, magy, magz, contact):
        return self.translationWrapper.forward( motion, edit_joint, frame, magx, magy, magz, contact)



