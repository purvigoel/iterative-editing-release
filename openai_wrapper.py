import openai
import time
import ast
import json
import numpy as np
import pose_mask
import torch
import sys
import traceback
from llm.assertions import *

messages = []

def query_gpt():
    MODEL = "gpt-4" #"gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )
    content = response.choices[0].message.content
    return content

def sequence_content(content, prompt2, append=True):
    if(append):
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": prompt2})
    else:
        messages[-2] = {"role": "assistant", "content": content}
        messages[-1] = {"role": "user", "content": prompt2}

def query_gpt_sequential(prompt_sequence):
        responses = []
        for i in range(0, len(prompt_sequence) - 1, 2):
            messages.append({"role": "user", "content": prompt_sequence[i]})
            messages.append({"role": "assistant", "content": prompt_sequence[i + 1]})
        messages.append({"role": "user", "content": prompt_sequence[-1]})
        content = query_gpt()
        responses.append(content)
        messages.clear()
        return responses[-1]

def response_to_code(responses, err_prompt_sequence, try_counter, logger = None, trylist=[]):
    if try_counter >= 3:
        print("Giving up")
        return None, -1
    responses_split = responses.split("\n")
    methods = []
    valid = True
    not_defined = False
    found_load_motion = False
    found_save_motion = False
    print(responses)
    counter = 0
    for response in responses_split:
        if "do_" in response and "(" in response and "undo" not in response:
            methods.append(response.strip())
        if "load_motion" in response:
            found_load_motion = True
        if "save_motion" in response:
            found_save_motion = True

    for method in methods:
        try:
            print(method)
            success, err = eval(method)
            if success < 0:
                err_prompt_sequence.append(responses)
                tb = traceback.format_exc()
                err_prompt_sequence.append(err)
                valid = False
                break
        except Exception as err:
            print("try except", err)
            err_prompt_sequence.append(responses)
            tb = traceback.format_exc()
            err_prompt_sequence.append(str(err))
            valid = False
            break

    found_methods = False
    if len(methods) > 0:
        found_methods = True
    elif len(methods) == 0 and found_load_motion and found_save_motion:
        found_methods = True
    else:
        print(methods)

    if not found_methods and not not_defined :
        print("Invalid Program")
        print(responses)
        err_prompt_sequence.append(responses)
        err_prompt_sequence.append("Please respond by editing your invalid program.")
        valid = False
    trylist.append(counter)
    if not valid:
        if logger:
            logger.log_error(responses)
        code, responses = query_model(err_prompt_sequence, err_prompt_sequence, try_counter + 1, logger,trylist)
        counter += 1
        #trylist.append(counter)
    else:
        code = responses
    return code, counter

prompt_sequence = []
def query_model(prompt, err_prompt_sequence, try_counter, logger = None, trylist=[]):
    print("querying model")
    responses = query_gpt_sequential(prompt)
    code , responses = response_to_code(responses, err_prompt_sequence, try_counter, logger, trylist)
    return code, responses

def read_progprompt( edit_instruction ):
    print(edit_instruction)
    with open("llm/prog_prompt3.py", "r") as f:
        lines = f.read()
        prompt_sequence.append("```python\n" + lines + "```")

def get_incontext():
    prompt_sequence[-1] += "# the person is walking. The right foot should be way higher.\n"
    prompt_sequence.append('''def step_way_higher():
        load_motion("motion_0")
        # the original motion is that the person is walking. The desired edit is that the right foot should be way higher.
        # the primary joints involved are the right foot, right knee, and right hip
        identified_joints = ["right_foot", "right_knee", "right_hip"]
        # the motion timing is relative to another joint: when the right foot is at its highest point, since that point needs to be adjusted. The desired edit is a change in pose, so it is not a verb.

        # move right foot up
        do_translate("right_foot", "up", time=when_joint("right_foot", "highest", is_verb=False))
        # flex hip to get the whole leg higher
        do_rotate("right_hip", "flex", time=when_joint("right_foot", "highest", is_verb=False))
        load_motion("motion_1")
    ''')

    prompt_sequence.append("# now swing your right arm out in the middle of the motion.\n")
    prompt_sequence.append('''def swing_arm_out():
        load_motion("motion_1")
        # the original motion is that the person is walking and was edited to get the right foot higher. The desired edit is to swing the right arm out in the middle of the motion."
        # the primary joints in volved are the right shoulder.
        identified_joints = ["right_shoulder"]
        # the motion timing is relative to the global motion: the middle of the motion"

        # abduct the right shoulder to swing the arm out to the side.
        do_rotate("right_shoulder", "abduct", time=at_global_moment("middle_of_motion", is_verb=True))

        save_motion("motion_2")
    '''
    )
    prompt_sequence.append("# your other arm too, at the same time.\n")
    prompt_sequence.append('''def swing_other_arm_out():
        load_motion("motion_2")
        # the original motion is that the person is walking and was edited to get the right foot higher and right arm out. The desired edit is to do the same with the other arm, at the same time."
        # the primary joints in volved are the right shoulder.
        identified_joints = ["left_shoulder"]
        # the motion timing is relative to the global motion: the middle of the motion"

        # abduct the left shoulder to swing the arm out to the side.
        do_rotate("left_shoulder", "abduct", time=at_global_moment("middle_of_motion", is_verb=True))

        save_motion("motion_3")
    '''
    )
    prompt_sequence.append("# swing it forward, not out.\n")
    prompt_sequence.append('''def swing_other_arm_out2():
        # I need to correct my editing program, so I load motion_2 to apply a new edit, not motion_3.
        load_motion("motion_2")
        # the original motion is that the person is walking and was edited to get the right foot higher and right arm out and left arm out. The desired edit is a clarification that the left arm should swing forward, not out. So we need to edit the previous motion and try again."
        # the primary joints in volved are the right shoulder.
        identified_joints = ["left_shoulder"]
        # the motion timing is relative to the global motion: the middle of the motion"

        # flex the left shoulder to swing the arm forward.
        do_rotate("left_shoulder", "flex", time=at_global_moment("middle_of_motion", is_verb=True))

        # save the corrected edit
        save_motion("motion_4")
    '''
    )
    
    prompt_sequence.append("# Do these arm motions at the end of the motion, not the middle.\n")
    prompt_sequence.append('''def move_kick_to_end():
        # I need to load the motion before I edited the arm swinging. That would be motion_1.
        load_motion("motion_1")

        # The original motion in this case is that the person is walking, and the desired edit is to swing the arms at the end.
        # the primary joints in volved are the left shoulder and right shoulder
        identified_joints = ["left_shoulder", "right_shoulder"]
        # the motion timing is relative to the global motion: end of the motion

        # abduct the right shoulder to preserve the previous edits.
        do_rotate("right_shoulder", "abduct", time=at_global_moment("end_of_motion", is_verb=True))
         # flex the left shoulder to swing the arm forward as described in the previous edits.
        do_rotate("left_shoulder", "flex", time=at_global_moment("end_of_motion", is_verb=True))

        save_motion("motion_5")
    '''
    )

    prompt_sequence.append("# Undo all the arm edits.\n")
    prompt_sequence.append('''def undo_arms():
        # I need to revert the edit to before I edited the arms. I can do that by loading the previous motion, motion_1.
        load_motion("motion_1")

        # Then I save it without any changes as motion_5.
        save_motion("motion_6")
    '''
    )
    prompt_sequence.append("Let's start from scratch, with a new motion, motion_0. Ready?")
    prompt_sequence.append("Yes!")


if len(sys.argv) > 1 and sys.argv[1] == "chatbot":
    read_progprompt("")
    get_incontext()
    while True:
        user_input = input("You: ")
        print("Chatbot:", user_input)
        prompt = user_input
        
        prompt_sequence.append("# " + prompt + "\n")
        error_prompt_sequence = prompt_sequence
        c, r = query_model(prompt_sequence, error_prompt_sequence, 0)
        prompt_sequence.append(c)
        print(c)


