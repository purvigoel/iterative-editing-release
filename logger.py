import os
import numpy as np

class Logger:
    def __init__(self):
        self.folder = True

    def init_folder(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder

    def log_error(self, string):
        self.init_folder(self.folder)
        with open(self.folder + "/error.txt", "a") as f:
            f.write(string)

    def log_txt(self, folder, filename, string):
        self.init_folder(folder)
        if string is None:
            string = "Gave up"
        with open(folder + "/" + filename + ".txt", "w") as f:
            f.write(string)

    def save_motion(self, folder, filename, motion):
        self.init_folder(folder)
        np.save(folder + "/" + filename + ".npy", motion)
