from actions import translate_joint, rotate_joint, relative_translate_joint, fix_joint
from timing import when_joint, at_global_moment, before_joint, as_joint, at_frame, get_current_frame
from speed import change_speed
from motion_io import load_motion, save_motion

relative_moments = ["highest", "lowest", "midrange_height", "furthest_from_body", "closest_to_body"]

global_moments = ["start_of_motion", "end_of_motion", "middle_of_motion", "entire_motion"]

translate_directions = ["forward", "backward", "up", "down"]
rotation_directions = ["abduct", "adduct", "extend", "flex"]
relative_translate_directions = ["towards", "above", "below", "in_front", "contact", "next_to"]

# IMMUTABLE: joints that move with rotation
joints_that_rotate = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_knee", "left_knee", "right_hip", "left_hip"]
# IMMUTABLE: joints that move with translation
joints_that_translate = ["right_foot", "left_foot", "right_hand", "left_hand", "waist"]

other_locations = ["ground"]

speeds = ["fast", "slow", "pause"]

# rotate a joint in a certain direction
def do_rotate(joint, direction, time):
    assert(joint in joints_that_rotate)
    assert(direction in rotation_directions)
    assert(time in relative_moments or time in global_moments)
    rotate_joint(joint, direction, time)

# translate a joint to a certain direction
def do_translate(joint, direction, time):
    assert(joint in joints_that_translate)
    assert(direction in translate_directions)
    assert(time in relative_moments or time in global_moments)
    translate_joint(joint, direction, time)

# Constrain a joint to a location between start_time and end_time
def do_relative_translate(jointA, location, direction, start_time, end_time):
    assert(jointA in joints_that_translate)
    assert(location in joints_that_translate or location in joints_that_lotate or location in other_locations)
    assert(direction in relative_translate_directions)
    assert(end_time in relative_moments or end_time in global_moments)
    assert(start_time in relative_moments or start_time in global_moments)
    relative_translate_joint(jointA, jointB, direction, start_time, end_time)

def do_change_speed(speed, start_time, end_time):
    assert(speed in speeds)
    assert(end_time in relative_moments or end_time in global_moments)
    assert(start_time in relative_moments or start_time in global_moments)
    change_speed(speed, start_time, end_time)

def do_fix_joint(joint, location, start_time, end_time)
    assert(joint in joints_that_translate)
    fix_joint(joint, location, start_time, end_time)

# the person is jumping. bring right knee to chest at the highest point of the jump
def right_knee_to_chest():
    # load the motion that needs to be edited
    load_motion("motion_0")

    # the original motion is that the person is jumping. the desired edit is bring right knee to chest.
    # the joints involved are the right knee and the right hip.
    identified_joints = ["right_knee", "right_hip"]
   
    # the motion timing is relative to another joint: when the waist is highest. The desired edit is a verb

    # bend right knee
    do_rotate("right_knee", "flex", time=when_joint("waist", "highest", is_verb=True))
    # flex the right hip to bring the knee higher
    do_rotate("right_hip", "flex", time=when_joint("waist", "highest", is_verb=True))

    # save edited motion
    save_motion("motion_1")

# the person is standing still. leap into the air at the start of the motion
def jump():
    # the original motion is that the person is standing still. The desired edit is leap into the air.
    # load the motion that needs to be edited
    load_motion("motion_0")

    # the primary joint involved is the entire body, summarized by the waist.
    identified_joints = ["waist"]
    # the motion timing is relative to a point in the global motion: the start of the motion. The desired edit is a verb.

    # the waist represents the whole body, and we move it upwards to simulate jumping
    do_translate("waist", "up", time=at_global_moment("start_of_motion", is_verb=True))
    
    # save edited motion
    save_motion("motion_1")

# the person is dancing. get lower to the ground the entire time
def squat():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is dancing. The desired edit is get lower to the ground.
    # the primary joint involved is the entire body, summarized by the waist.
    identified_joints = ["waist"]
    # the motion timing is relative to points in the global motion: the entire motion. The desired edit is not a verb.
    # the waist needs to be closer to the ground
    do_translate("waist", "down", time=at_global_moment("entire_motion", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is hopping up and down. raise arms out to the side as you jump
def raise_arms_out_to_side():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is hopping up and down. The desired edit is raise arms out to the side as you jump.
    # the primary joints involved are the right shoulder and left shoulder.
    identified_joints = ["right_shoulder", "left_shoulder"]
    # the motion timing is relative to an action in the entire motion, as the character jumps. The desired edit is a verb.

    # move right arm out to the side
    do_rotate("right_shoulder", "abduct", time=as_joint("waist", "highest", is_verb=True))
    # move left arm out to the side
    do_rotate("left_shoulder", "abduct", time=as_joint("waist", "highest", is_verb=True))

    # save edited motion
    save_motion("motion_1")

# the person is squatting. Get your right hand above your head at the end of the motion.
def right_hand_above_head():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is squatting. The desired edit is get your right hand above your head.
    # the primary joints involved are the right hand.
    identified_joints = ["right_hand"]
    # the motion timing is relative to a point in the global motion: the end of the  motion. The desired edit is not a verb.
    
    # this is a relative translation involving moving the the right hand above the head.
    do_relative_translate("right_hand", "head", "above", start_time=at_global_moment("end_of_motion", is_verb=False), end_time=at_global_moment("end_of_motion", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is squatting. Get your right hand above your head at frame 40.
def right_hand_above_head():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is squatting. The desired edit is get your right hand above your head.
    # the primary joints involved are the right hand.
    identified_joints = ["right_hand"]
    # the motion timing is relative to a point in the global motion: frame 40. The desired edit is not a verb.

    # this is a relative translation involving moving the the right hand above the head.
    do_relative_translate("right_hand", "head", "above", start_time=at_frame("40", is_verb=False), end_time=at_frame("40", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is standing. Reach for your right hand with your left when your left hand is at its highest.
def reach_for_right_hand():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is standing. The desired edit is reach for your right hand with your left.
    # the primary joints involved are the left hand.
    identified_joints = ["right_hand", "left_hand"]
    # the motion timing is relative to another joint: when the left hand is highest. The desired edit is a verb.

    # this is a relative translation involving moving the left hand towards the right hand.
    do_relative_translate("left_hand", "right_hand", "towards", start_time=when_joint("left_hand", "highest", is_verb=True), end_time=when_joint("left_hand", "highest", is_verb=True))

    # save edited motion
    save_motion("motion_1")

# the person is swinging a racket. before you swing, your right hand should be blocking your face.
def block_face_before_swing():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is swinging a racket. The desired edit is block your face with your right hand before the swing.
    # the primary joints involved are the right hand.
    identified_joints = ["right_hand"]
    # the motion timing is relative to another joint: before the right hand swings and extends. The desired edit is not a verb.
    # this is a relative translation involving moving the right hand infront of the face.
    do_relative_translate("right_hand", "head", "in_front", start_time=before_joint("right_hand", "furthest_from_body", is_verb=False), end_time=before_joint("right_hand", "furthest_from_body", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is swinging a racket. as you swing, your right hand should be blocking your face.
def block_face_during_swing():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is swinging a racket. The desired edit is block your face with your right hand as you swing.
    # the primary joints involved are the right hand.
    identified_joints = ["right_hand"]
    # the motion timing is relative to another joint's action: as the right hand swings and extends. The desired edit is not a verb.
    
    # this is a relative translation involving moving the right hand infront of the face.
    do_relative_translate("right_hand", "head", "in_front", start_time=as_joint("right_hand", "furthest_from_body", is_verb=False), end_time=as_joint("right_hand", "furthest_from_body", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is punching the right hand into the air. The punch should be a lot higher.
def punch_a_lot_higher():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is punching the right hand into the air. The desired edit is the punch should be a lot higher.
    # the primary joints involved are the right hand, right elbow, and right shoulder.
    identified_joints = ["right_hand", "right_elbow", "right_shoulder"]
    # the motion timing is relative to another joint: when the punch is at its highest point, since that point needs to be adjusted. The desired edit is a change in pose, so it is not a verb.
    # move right hand up
    do_rotate("right_elbow", "extend",  time=when_joint("right_hand", "highest", is_verb=False))
    # flex the shoulder to get the arm higher
    do_rotate("right_shoulder", "flex", time=when_joint("right_hand", "highest", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is punching the right hand into the air. The punch should be slightly higher.
def punch_slightly_higher():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is punching the right hand into the air. The desired edit is the punch should be slightly higher.
    # the primary joints involved are the right hand
    identified_joints = ["right_hand"]
    # the motion timing is relative to another joint: when the punch is at its highest point, since that point needs to be adjusted. The desired edit is a change in pose, so it is not a verb.
    # move right hand up
    do_translate("right_hand", "up", time=when_joint("right_hand", "highest", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is walking. The right foot should be way higher.
def step_way_higher():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is walking. The desired edit is that the right foot should be way higher.
    # the primary joints involved are the right foot, right knee, and right hip
    identified_joints = ["right_foot", "right_knee", "right_hip"]
    # the motion timing is relative to another joint: when the right foot is at its highest point, since that point needs to be adjusted. The desired edit is a change in pose, so it is not a verb.
    # move right foot up
    # flex hip to get the whole leg higher
    do_rotate("right_hip", "flex", time=when_joint("right_foot", "highest", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is walking. The right foot should be a little higher.
def step_a_little_higher():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is walking. The desired edit is that the right foot should be a little higher.
    # the primary joints involved are the right foot.
    identified_joints = ["right_foot"]
    # the motion timing is relative to another joint: when the right foot is at its highest point, since that point needs to be adjusted. The desired edit is a change in pose, so it is not a verb.
    # move right foot up
    do_translate("right_foot", "up", time=when_joint("right_foot", "highest", is_verb=False))

    # save edited motion
    save_motion("motion_1")

# the person is dancing. As you raise your arms, do a jump.
def add_a_jump():
    # load motion
    load_motion("motion_0")
    # the original motion is that the person is dancing. The desired edit is as you raise your arms do a jump.
    # the primary joints involved are the waist
    identified_joints = ["waist"]
    # the motion timing is relative to an action in the entire motion, as the arm raise. When referring to relative actions.

    do_translate("waist", "up", time=as_joint("right_hand","highest",is_verb=True))

    save_motion("motion_1")

# the person is kicking with the right foot. Do that kick faster.
def speed_up_kick():
    # load motion
    load_motion("motion_0")

    # the original motion is that a person is kicking. The desired edit is to kick faster.
    # the speed is faster.
    # the motion timing is to start speeding up at the start of the kick and stop speeding up at the top of the kick.
    do_change_speed("fast", start_time=when_joint("right_foot", "lowest", is_verb=False), end_time=when_joint("right_foot", "highest", is_verb=False))

    save_motion("motion_1")

# the person is walking. Lift the foot right here.
def lift_foot_at_moment():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is walking. The desired edit is that the right foot should be a little higher.
    # the primary joints involved are the right foot.
    identified_joints = ["right_foot"]

    # the motion timing is relative to another joint: when the right foot is at its highest point, since that point needs to be adjusted. The desired edit is a change in pose, so it is not a verb.
    # move right foot up
    do_translate("right_foot", "up", time=at_frame(get_current_frame(), is_verb=True))

    # save edited motion
    save_motion("motion_1")

# the person is walking. Okay, right now, jump into the air
def lift_foot_at_moment():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is dancing. The desired edit is as you raise your arms do a jump.
    # the primary joints involved are the waist
    identified_joints = ["waist"]
    # the motion timing is relative to an action in the entire motion, as the arm raise. When referring to relative actions.

    do_translate("waist", "up", time=at_frame(get_current_frame(), is_verb=True))

    save_motion("motion_1")

# the person is walking. Stop for a second at frame 50
def lift_foot_at_moment():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is dancing. The desired edit is as you raise your arms do a jump.
    # the primary joints involved are the waist
    identified_joints = ["waist"]
    # the motion timing is relative to an action in the entire motion, as the arm raise. When referring to relative actions.

    do_change_speed("pause", start_time=at_frame("50", is_verb=False), end_time=at_frame("50", is_verb=False))

    save_motion("motion_1")

# the person is walking. Keep your right hand on your hip the whole time.
def hand_on_hip():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is walking. The desired edit is to keep the hand on the hip the whole time.
    # the primary joits involved are the right hand.
    identified_joints = ["right_hand"]

    # the motion timing is to start at the start of the motion and end at the end of the motion. I want  to describe the motion of the right hand with respect to another joint.
    do_relative_translate("right_hand", "right_hip", "contact", start_time=at_global_moment("start_of_motion", is_verb=False), end_time=at_global_moment("end_of_motion", is_verb=False))

    save_motion("motion_1")

# the person is standing and raises the right leg. Raise the left foot with the right one.
def raise_hands_same_time():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is standing and raises the right leg. The desired edit is to raise the left leg with the right one.
    # the primary joints in volved are the left foot. I could use the whole left leg, but I want to apply a positional constraint  to describe its motion with respect to another joint.
    identified_joints = ["left_foot"]

    # the motion timing is to start when the right hand is lowest and end when it is highest. To synchronize the hands, I can use a relative translation.
    do_relative_translate("left_foot", "right_foot", "next_to", start_time=when_joint("right_foot", "lowest", is_verb=False), end_time=when_joint("right_foot", "highest", is_verb=False))

    save_motion("motion_1")

# the person punches the right arm. Do that with your left arm too.
def raise_hands_same_time():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is standing and punches the right arm. The desired edit is do that with your left arm too.
    # the primary joints in volved are the left hand. I could use the whole left arm, but I want to apply a positional constraint  to describe its motion with respect to another joint.
    identified_joints = ["left_hand"]

    # the motion timing is to start when the right hand is closest and end when it furthest from the body, to simulate a punch. To synchronize the hands, I can use a relative translation.
    do_relative_translate("left_hand", "right_hand", "next_to", start_time=when_joint("right_hand", "closest_to_body", is_verb=False), end_time=when_joint("right_hand", "furthest_from_body", is_verb=False))
    
    save_motion("motion_1")

# the person throws a basketball with their right arm. Use your left hand too.
def throw_ball_with_both_hands():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is throwing with the right arm. The desired edit is do that with your left arm too.
    # the primary joints in volved are the left hand. I could use the whole left arm, but I want to apply a positional constraint to describe its motion with respect to another joint.
    identified_joints = ["left_hand"]

    # the motion timing is to start when the right hand is lowest and end when it is highest, to simulate throwing a basketball. To synchronize the hands, I can use a relative translation.
    do_relative_translate("left_hand", "right_hand", "next_to", start_time=when_joint("right_hand", "lowest", is_verb=False), end_time=when_joint("right_hand", "highest", is_verb=False))

    save_motion("motion_1")
