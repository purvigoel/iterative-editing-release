import os
from tqdm import tqdm
import shutil
import os.path as osp
import joblib
import ipdb
from collections import defaultdict
#from nemo.utils.misc_utils import *
from hmr.renderer import Renderer
from hmr.smpl import SMPL
from hmr import hmr_config
import cv2
# VIBE related
#from VIBE.lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR, H36M_DIR
#from VIBE.lib.models import spin
#from VIBE.lib.data_utils.kp_utils import *
#from lib.data_utils.feature_extractor import extract_features

import matplotlib.pylab as plt
import argparse

#from nemo.utils.pose_utils import rigid_transform_3D, apply_rigid_transform
