import numpy as np

class SaveStack:
    def __init__(self):
        self.stack = {}

    def load_motion(self, name):
        return self.stack[name]
    
    def save_motion(self, name, motion):
        self.stack[name] = motion
