import numpy as np

SMPLH_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'right_index1'
]
pose_dict = {}
for ind, name in enumerate(SMPLH_JOINT_NAMES):
    pose_dict[name] = ind
pose_dict["right_hand"] = 21
pose_dict["left_hand"] = 20
pose_dict["right_leg"] = pose_dict["right_foot"]
pose_dict["left_leg"] = pose_dict["left_foot"]
pose_dict["right_arm"] = pose_dict["right_hand"]
pose_dict["left_arm"] = pose_dict["left_hand"]
pose_dict["body"] = pose_dict["pelvis"]
pose_dict["waist"] = pose_dict["pelvis"]
pose_dict["right_foot"] = pose_dict["right_ankle"]
pose_dict["left_foot"] = pose_dict["left_ankle"]

leg_joints = [1, 2, 4, 5, 7, 8, 10, 11]

kinematic_chains = [
            [0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [22, 20, 18, 16, 13, 9], [23, 21, 19, 17, 14, 9]
        ]


keep = ["left_hip", "right_hip", "left_collar", "right_collar", "left_foot", "left_knee", "left_ankle", "right_foot", "right_ankle", "right_knee"]

#keep = ["pelvis", "left_ankle", "right_hip", "right_knee", "right_ankle", "right_foot", "left_foot", "right_elbow", "right_collar", "right_wrist", "left_elbow", "left_collar", "left_wrist", "left_index1", "right_index1"]

lower_body = ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot']
left_hand = ["left_wrist"]
left_arm = ['left_collar', 'left_shoulder', 'left_elbow', "left_index1"]
right_arm = ['spine', 'right_collar', 'right_shoulder', 'right_elbow', 'right_index1']
left_foot_pos = [10]

right_arm_joints = [14, 17, 19, 21, 23]
right_arm_angles = []
for i in right_arm_joints:
    for x in range(6):
        right_arm_angles.append(i * 6 + x)

right_arm_joints_kc = [14, 17, 19, 21, 23, 9 , 6, 3]
right_arm_angles_kc = []
for i in right_arm_joints_kc:
    for x in range(6):
        right_arm_angles_kc.append(i * 6 + x)

right_arm_joints2 = [14, 17, 19]
right_arm_positions = []
for i in right_arm_joints2:
    for x in range(3):
        right_arm_positions.append(i * 3 + x + 164)

right_arm_joints2 = [ 17, 19, 21, 23]
right_arm_angles2 = []
for i in right_arm_joints2:
    for x in range(6):
        right_arm_angles2.append(i * 6 + x)

right_arm_joints3 = [ 17, 19, 21, 23]
right_arm_angles3 = []
for i in right_arm_joints3:
    for x in range(6):
        right_arm_angles3.append(i * 6 + x)

left_arm_joints = [13, 16, 18, 20, 22]
left_arm_angles = []
for i in left_arm_joints:
    for x in range(6):
        left_arm_angles.append(i * 6 + x)

left_arm_joints2 = [13, 16, 18, 20]
left_arm_positions = []
for i in left_arm_joints2:
    for x in range(3):
        left_arm_positions.append(i * 3 + x + 164)

left_arm_joints_kc = [13, 16, 18, 20, 22, 9 , 6, 3]
left_arm_angles_kc = []
for i in left_arm_joints_kc:
    for x in range(6):
        left_arm_angles_kc.append(i * 6 + x)

left_arm_joints2 = [ 16, 18, 20, 22]
left_arm_angles2 = []
for i in left_arm_joints2:
    for x in range(6):
        left_arm_angles2.append(i * 6 + x)

left_arm_joints3 = [16, 18, 20, 22]
left_arm_angles3 = []
for i in left_arm_joints3:
    for x in range(6):
        left_arm_angles3.append(i * 6 + x)

left_leg_joints = [ 1, 4, 7, 10]
left_leg_angles = []
for i in left_leg_joints:
    for x in range(6):
        left_leg_angles.append(i * 6 + x)

left_leg_joints2 = [1, 4, 7]
left_leg_positions = []
for i in left_leg_joints2:
    for x in range(3):
        left_leg_positions.append(i * 3 + x + 164)

left_leg_joints2 = [ 4, 7, 10]
left_leg_angles2 = []
for i in left_leg_joints2:
    for x in range(6):
        left_leg_angles2.append(i * 6 + x)

right_leg_joints = [ 2, 5, 8, 11]
right_leg_angles = []
for i in right_leg_joints:
    for x in range(6):
        right_leg_angles.append(i * 6 + x)

right_leg_joints2 = [2, 5, 8]
right_leg_positions = []
for i in right_leg_joints2:
    for x in range(3):
        right_leg_positions.append(i * 3 + x + 164)

right_leg_joints2 = [ 5, 8, 11]
right_leg_angles2 = []
for i in right_leg_joints2:
    for x in range(6):
        right_leg_angles2.append(i * 6 + x)

spine_joints = [0, 3, 6, 9, 12, 15]
spine_angles = []
for i in spine_joints:
    for x in range(6):
        spine_angles.append(i * 6 + x)

spine_joints_nr = [ 3, 6, 9, 12, 15]
spine_angles_nr = []
for i in spine_joints:
    for x in range(6):
        spine_angles_nr.append(i * 6 + x)

knees = ["left_knee", "right_knee"]
left_knee = ["left_foot", "right_foot"]
left_leg = [ "left_hip", "left_ankle", "left_knee", "right_hip", "right_ankle", "right_knee"]

knees = left_leg

opposite = ["left_foot", "right_wrist"]

#pose6d -> 144 (24 x 6)
#translation -> (1 x 6)
#foot contact mask -> 154 (1 x 4)
#beta params -> 164 (1 x 10)
#positions -> 236 (24 x 3)
#velocity -> 308 (24 * 3)

NUM_SMPLH_JOINTS = len(SMPLH_JOINT_NAMES)

num_features = 308

feature_mask = np.zeros((308))

def keep_upper_body_joint_angles():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in lower_body and i != 0:
            for j in range(6):
                mask.append(0)    
        else:
            for j in range(6):
                mask.append(1)
    return mask

def erase_upper_body_joint_angles():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in lower_body:
            for j in range(6):
                mask.append(1)
        else:
            for j in range(6): 
                mask.append(0)
    return mask

def erase_lower_body_joint_angles():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] not in lower_body:
            for j in range(6):
                mask.append(1)
        else:
            for j in range(6):
                mask.append(0)
    return mask

def keep_left_arm_joint_angles():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] not in left_arm and i != 0:
            for j in range(6):
                mask.append(0)
        else:
            for j in range(6):
                mask.append(1)
    return mask    

def keep_right_arm_joint_angles():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] not in right_arm and i != 0:
            for j in range(6):
                mask.append(0)
        else:
            for j in range(6):
                mask.append(1)
    return mask

def erase_joint_angles():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        for j in range(6):
            mask.append(0)
    return mask

def keep_joint_angles():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        for j in range(6):
            mask.append(1)
    return mask

def erase_joint_angles_except_root():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if i == 0:
            for j in range(6):
                mask.append(1)
        else:
            for j in range(6):
                mask.append(0)
    return mask

def keep_joint_angles_not_pelvis():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if i == 0:
            for j in range(6):
                mask.append(0)
        else:
            for j in range(6):
                mask.append(1)
    return mask

def keep_translation():
    mask = []
    for j in range(6):
        mask.append(1)
    return mask

def erase_translation():
    mask = []
    for j in range(6):
        mask.append(0)
    return mask

def keep_foot_contact():
    mask = []
    for j in range(4):
        mask.append(1)
    return mask

def erase_foot_contact():
    mask = []
    for j in range(4):
         mask.append(0)
    return mask

def keep_beta_params():
    mask = []
    for j in range(10):
        mask.append(1)
    return mask

def keep_positions():
    mask = []
    for j in range(NUM_SMPLH_JOINTS * 3):
        mask.append(1)
    return mask

def erase_positions():
    mask = []
    for j in range(NUM_SMPLH_JOINTS * 3):
        mask.append(0)
    return mask

def keep_upper_body_positions():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in lower_body:
            for j in range(3):
                mask.append(0)
        else:
            for j in range(3):
                mask.append(1)
    return mask

def erase_upper_body_joint_positions():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in lower_body:
            for j in range(3):
                mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_upper_body_joint_positions_keep_left_hand():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in lower_body or SMPLH_JOINT_NAMES[i] in left_hand:
            for j in range(3):
                mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_joint_positions_keep_left_hand():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in left_hand:
            for j in range(3):
                mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_left_arm_not_hand():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in left_hand or SMPLH_JOINT_NAMES[i] not in left_arm:
            for j in range(3):
                mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_left_leg_not_knee():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
  
        if SMPLH_JOINT_NAMES[i] in left_knee:
            mask.append(1)
            mask.append(1)
            mask.append(1)
        elif SMPLH_JOINT_NAMES[i] not in left_leg:
            for j in range(3):
                mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_positions_not_knee():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):

        if SMPLH_JOINT_NAMES[i] in left_knee :
            mask.append(1)
            mask.append(1)
            mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask
#keep
def erase_positions_not_square():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):

        if SMPLH_JOINT_NAMES[i] in keep :
            mask.append(1)
            mask.append(1)
            mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_joint_angles_not_square():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):

        if SMPLH_JOINT_NAMES[i] in keep :
            for j in range(6):
                mask.append(1)
        else:
            for j in range(6):
                mask.append(0)
    return mask

def erase_positions_not_knee_y():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):

        if SMPLH_JOINT_NAMES[i] in left_knee:
            mask.append(0)
            mask.append(1)
            mask.append(0)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_velocity_not_knee():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):

        if SMPLH_JOINT_NAMES[i] in left_knee:
            mask.append(1)
            mask.append(1)
            mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_positions_not_knees():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):

        if SMPLH_JOINT_NAMES[i] in knees:
            mask.append(1)
            mask.append(1)
            mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_positions_not_hand():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):

        if SMPLH_JOINT_NAMES[i] in left_hand:
            mask.append(1)
            mask.append(1)
            mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def erase_positions_not_opposite():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):

        if SMPLH_JOINT_NAMES[i] in opposite:
            mask.append(1)
            mask.append(1)
            mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def keep_velocity():
    mask = []
    for j in range(NUM_SMPLH_JOINTS * 3):
        mask.append(1)
    return mask


def keep_upper_body_velocity():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in lower_body:
            for j in range(3):
                mask.append(0)
        else:
            for j in range(3):
                mask.append(1)
    return mask

def erase_upper_body_joint_velocity():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in lower_body:
            for j in range(3):
                mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask

def keep_velocity():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        for j in range(3):
            mask.append(1)
    return mask

def erase_velocity():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        for j in range(3):
            mask.append(0)
    return mask

def erase_velocity_not_knees():
    mask = []
    for i in range(len(SMPLH_JOINT_NAMES)):
        if SMPLH_JOINT_NAMES[i] in knees:
            for j in range(3):
                mask.append(1)
        else:
            for j in range(3):
                mask.append(0)
    return mask



#pose6d -> 144 (24 x 6)
#translation -> (1 x 6)
#foot contact mask -> 154 (1 x 4)
#beta params -> 164 (1 x 10)
#positions -> 236 (24 x 3)
#velocity -> 308 (24 * 3)

def upper_body_mask():
    mask = []
    mask.extend(keep_upper_body_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_positions())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def upper_body_mask_vel():
    mask = []
    mask.extend(keep_upper_body_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_positions())
    mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def joint_angle_mask():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(keep_translation())
    mask.extend(keep_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    #mask.extend(keep_velocity())
    mask = np.array(mask)
    return mask

def joint_angle_mask_keepgr():
    mask = []
    mask.extend(erase_joint_angles_except_root())
    mask.extend(keep_translation())
    mask.extend(keep_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    #mask.extend(keep_velocity())
    mask = np.array(mask)
    return mask

def joint_angle_mask_erase_all():
    mask = []
    mask.extend(erase_joint_angles_except_root())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def joint_angle_mask_no_vel():
    mask = []
    mask.extend(erase_joint_angles_except_root())
    mask.extend(keep_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def left_arm_mask():
    mask = []
    mask.extend(keep_left_arm_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_positions())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def joint_angle_mask_upper_body_notranslation():
    mask = []
    mask.extend(keep_upper_body_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def joint_angle_mask_upper_body():
    mask = []
    mask.extend(keep_upper_body_joint_angles())
    mask.extend(keep_translation())
    mask.extend(keep_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    #mask.extend(keep_velocity())
    mask = np.array(mask)
    return mask

def joint_angle_mask_right_arm_no_translation():
    mask = []
    mask.extend(keep_right_arm_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def joint_angle_mask_left_arm_no_translation():
    mask = []
    mask.extend(keep_left_arm_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def ik_lefthand_test():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_joint_positions_keep_left_hand())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

def ik_lefthand_test2():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_positions_not_hand())
    #mask.extend(erase_velocity())
    mask = np.array(mask)
    return mask

# erase_left_leg_not_knee

def ik_leftknee_test():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    #mask.extend(erase_left_leg_not_knee())
    mask.extend(erase_positions_not_knee())
    #mask.extend(erase_positions_not_square())
    #mask.extend(keep_positions())
    mask = np.array(mask)
    return mask

def ik_leftknee_test_rel():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(keep_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_positions_not_knee())
    mask = np.array(mask)
    return mask

def ik_knee_test():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    #mask.extend(erase_left_leg_not_knee())
    mask.extend(erase_positions_not_knees())
    mask = np.array(mask)
    return mask

def ik_knee_test_vel():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_positions_not_knees())
    mask.extend( erase_velocity() )
    mask = np.array(mask)
    return mask

def ik_opposite_test():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    #mask.extend(erase_left_leg_not_knee())
    mask.extend(erase_positions_not_opposite())
    mask = np.array(mask)
    return mask

def oppik_delete_pelvis_pos():
    mask = []
    mask.extend(keep_joint_angles())
    mask.extend(erase_translation())
    mask.extend(keep_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_positions())
    masp = np.array(mask)
    return mask

def ik_delete():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    mask = np.array(mask)
    return mask

def ik_vel():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(keep_translation())
    mask.extend(keep_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    mask.extend(keep_velocity())
    mask = np.array(mask)
    return mask

def ik_vel_rel():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(keep_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(keep_positions())
    mask = np.array(mask)
    return mask

def ik_vel2():
    mask = []
    mask.extend(erase_joint_angles())
    mask.extend(erase_translation())
    mask.extend(erase_foot_contact())
    mask.extend(keep_beta_params())
    mask.extend(erase_positions_not_knees())
    mask.extend(erase_velocity_not_knees())
    mask = np.array(mask)
    return mask

def delete_everything():
    return np.zeros((236))

SMPL_UPPERBODY_MASK = np.array( upper_body_mask() )
#SMPL_IKSOLVER_MASK = np.array( joint_angle_mask() )
SMPL_LEFTARM_MASK = np.array( left_arm_mask() )
SMPL_IKSOLVER_MASK = np.array(ik_delete())
SMPL_IKSOLVER_GR_MASK = np.array(joint_angle_mask_keepgr())
SMPL_IKSOLVER_NOVEL_MASK = np.array(joint_angle_mask_no_vel())
SMPL_IKSOLVER_UB_MASK = np.array(joint_angle_mask_upper_body_notranslation())
SMPL_DELETE_MASK = np.array(delete_everything())
SMPL_IKSOLVER_RIGHTARM_MASK = np.array(joint_angle_mask_right_arm_no_translation())
SMPL_IKSOLVER_LEFTARM_MASK = np.array(joint_angle_mask_left_arm_no_translation())
SMPL_IKSOLVER_ALL_MASK = np.array(joint_angle_mask_erase_all())
SMPL_IKSOLVER_LEFTHAND_MASK = np.array(ik_lefthand_test2())
SMPL_IKSOLVER_LEFTKNEE_MASK = np.array(ik_leftknee_test())
SMPL_IKSOLVER_KNEE_MASK = np.array(ik_knee_test())
SMPL_IKSOLVER_OPP_MASK = np.array(ik_opposite_test())
SMPL_IKSOLVER_REL_MASK = np.array(ik_vel_rel())
SMPL_IKSOLVER_KNEE_REL_MASK = np.array(ik_leftknee_test_rel())
SMPL_DELETE_ROOT = np.array(oppik_delete_pelvis_pos())

SMPL_VEL_UPPERBODY_MASK = np.array(upper_body_mask_vel())
SMPL_VEL_KNEE_MASK = np.array(ik_knee_test_vel()) 
SMPL_VEL_IKSOLVER_MASK = np.array(ik_vel())
