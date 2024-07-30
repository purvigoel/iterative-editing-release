
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import joblib
import colorsys
import numpy as np
from tqdm import tqdm

from lib.utils.renderer import Renderer

from lib.utils.demo_utils import (
    prepare_rendering_results,
)

vibe_results = joblib.load('output/jackson_tennis/jacksion-tennis-orig-trimmed/vibe_output.pkl')
    
orig_width = 1080 
orig_height = 1920
num_frames = 303
image_folder = '/tmp/jacksion-tennis-orig-trimmed_mp4'
# import ipdb; ipdb.set_trace()

renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)

output_img_folder = f'{image_folder}_output'
os.makedirs(output_img_folder, exist_ok=True)

print(f'Rendering output video, writing frames to {output_img_folder}')

# prepare results for rendering
frame_results = prepare_rendering_results(vibe_results, num_frames)
mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

image_file_names = sorted([
    os.path.join(image_folder, x)
    for x in os.listdir(image_folder)
    if x.endswith('.png') or x.endswith('.jpg')
])

os.makedirs('locked_cam', exist_ok=True)
for frame_idx in tqdm(range(len(image_file_names))):
    img_fname = image_file_names[frame_idx]

    img = cv2.imread(img_fname)

    for person_id, person_data in frame_results[frame_idx].items():
        frame_verts = person_data['verts']
        frame_cam = person_data['cam']

        mc = mesh_color[person_id]

        mesh_filename = None

        # Remove camera translation
        frame_cam[2] = 0 
        frame_cam[3] = 0

        nimg = renderer.render(
            img,
            frame_verts,
            cam=frame_cam,
            color=mc,
            mesh_filename=mesh_filename,
        )
        cv2.imwrite(f'locked_cam/{frame_idx}.png', nimg)
"""
ffmpeg -y -threads 16 -i locked_cam/%d.png -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -v error locked_cam.mp4
"""

# # Play with Camera
# os.makedirs('moving_imgs', exist_ok=True)
# for dim in [2,3]:
#     for offset in np.linspace(-3, 3, 7):
#         moved_cam = np.array(frame_cam).copy()
#         moved_cam[dim] += offset
#         nimg = renderer.render(
#             img,
#             frame_verts,
#             cam=moved_cam,
#             color=mc,
#             mesh_filename=mesh_filename,
#         )
        # cv2.imwrite(f'moving_imgs/dim{dim}_offset_{offset}.png', nimg)