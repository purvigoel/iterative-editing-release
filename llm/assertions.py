
relative_moments = ["highest", "lowest", "midrange_height", "furthest_from_body"]
global_moments = ["start_of_motion", "end_of_motion", "middle_of_motion", "entire_motion"]

translate_directions = ["forward", "backward", "up", "down"]
rotation_directions = ["abduct", "adduct", "extend", "flex"]
relative_translate_directions = ["towards", "above", "below", "in_front"]

# joints that move with rotation
rotate_joints = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_knee", "left_knee", "right_hip", "left_hip"]
# joints that move with translation
translate_joints = ["right_foot", "left_foot", "right_hand", "left_hand", "waist"]

speed = ["fast", "slow", "pause"]

def do_rotate(joint, direction, time):
    if joint not in rotate_joints:
        return -1, "The " + joint + " cannot move with rotation. Can you provide a new program?"
    if direction not in rotation_directions:
        return -1, "The joint cannot move in " + direction
    return 0, ""

    assert(joint in rotate_joints)
    assert(direction in rotation_directions)

def do_translate(joint, direction, time):
    if joint not in translate_joints:
        return -1, "The " + joint + " cannot move with translation. Can you provide a new program?"
    if direction not in translate_directions:
        return -1, "The joint cannot move in " + direction
    return 0, ""
    assert(joint in translate_joints)
    assert(direction in translate_directions)

def do_relative_translate(jointA, location, direction, start_time, end_time):
    if jointA not in translate_joints:
        return -1, "The " + jointA + " cannot move with translation. Can you provide a new program?"
    
    if location not in translate_joints and location not in rotate_joints and location not in ["ground", "head"]:
        return -1, location + " is not an available location to place " + location + ". Can you provide a new program?"
    return 0, ""
    assert(jointA in translate_joints)
    #assert(jointB in translate_joints)
    assert(direction in relative_translate_directions)

def do_change_speed(direction, start_time, end_time):
    if direction not in speed:
        return -1, "The body cannot change speed by " + direc
    return 0, ""

def do_fix_joint(joint, location, start_time, end_time):
    if joint not in translate_joints:
        return -1, joint + " is not an available joint. Can you provide a new program?"
    if location not in translate_joints and location not in rotate_joints and location not in ["ground"]:
        return -1, location + " is not an available location to place " + joint + ". Can you provide a new program?"
    return 0, ""

def when_joint(drop1, drop2, is_verb=None):
    pass

def at_global_moment(drop1, is_verb=None):
    pass

def before_joint(drop1, drop2, is_verb=None):
    pass

def at_frame(drop1, is_verb=None):
    pass

def get_current_frame():
    pass

def as_joint(drop1, drop2, is_verb=None):
    if drop2 not in relative_moments:
        return -1, drop2 + " is not a relative moment. Can you provide a new program?"
    return 0, ""

