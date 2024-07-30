import numpy as np

def params_waist_adjust(params):
    direction = params["direction"]
    joint = params["joint"]
    if direction == "up":
        degrees = 0.05 # 0.25
    elif direction == "down":
        degrees = -0.15
        params["direction"] = "up"
    elif direction == "backward":
        degrees = -0.15
        params["direction"] = "backward"
    params["degrees"] = degrees
    return params

def params_rotation_adjust(params):
    direction = params["direction"]
    joint = params["joint"]
    amount = params["degrees"]
    frame = params["frame"]
    if direction == "adduct":
        direction = "abduct"
        amount = amount * -1
    elif direction == "abduct":
        pass
    elif direction == "extend" and joint == "shoulder":
        direction = "flex"
        amount = amount * -1
    elif direction == "extend" and joint == "hip":
        pass
        #direction = "flex"
        #amount = amount * -1
    elif direction == "flex" and joint == "hip":
        pass
    elif direction == "flex" and joint == "shoulder":
        pass
    elif direction == "extend" and joint == "knee":
        direction = "forward"
        amount *=-1
    #elif joint == "elbow":
    #    #direction = "flex"
    #    amount *= -1
    '''
    if frame != "entire_motion":
        window = 15
        if frame - window <= 1:
            window = frame - 1
        hi_window = 15
        if frame + hi_window >= 59:
            hi_window = 59 - frame
    '''
    print(params)
    params["direction"] = direction
    params["joint"] = joint
    params["degrees"] = amount
    params["frame"] = frame
    return params

def params_translation_adjust(params):
    direction = params["direction"]
    limb = params["joint"]
    frame = params["frame"]

    
    if direction == "aggregate":
        magnitude = magnitude
    elif direction == "out" and (limb == "right_foot" or limb == "right_hand"):
        magnitude = [-0.15, 0.0, 0.0]
    elif direction == "out":
        magnitude = [0.15, 0.0, 0.0]
    elif direction == "in" and (limb == "right_foot" or limb == "right_hand"):
        magnitude = [0.25, 0.0, 0.0]
    elif direction == "in":
        magnitude = [-0.25, 0.0, 0.0]
    elif direction == "forward":
        magnitude = [0.0, 0.0, 0.15]
    elif direction == "backward":
        magnitude = [0.0, 0.0, -0.25]
    elif direction == "up":
        magnitude = [0.0, 0.25, 0.0]
    elif direction == "down":
        magnitude = [0.0, -0.25, 0.0]
    elif direction == "in_front":
        magnitude = [0.0, 0.0, 0.25]
    else:
        magnitude = [0.0, 0.0, 0.0]

    '''
    window = 15
    if frame - window <= 1:
        window = frame - 1
    hi_window = 15
    if frame + hi_window >= 59:
        hi_window = 59 - frame
    '''
    params["magx"] = magnitude[0]
    params["magy"] = magnitude[1]
    params["magz"] = magnitude[2]
    return params
