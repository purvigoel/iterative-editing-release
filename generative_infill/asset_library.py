import numpy as np
import torch

asset_library = {}

asset_library[3] = torch.tensor(np.load("asset_library/3.npy"))
asset_library["3_desc"] = "A person kicks the right leg, then the left."
asset_library[2] = torch.tensor(np.load("asset_library/2.npy"))
asset_library["2_desc"] = "A person jumps."
asset_library[1] = torch.tensor(np.load("asset_library/1.npy"))
asset_library["1_desc"] = "A person is standing."

