import os 
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
print(os.environ['PYOPENGL_PLATFORM'])

from lib.utils.renderer import Renderer

# ========= Render results as a single video ========= #
renderer = Renderer(resolution=(100, 100), orig_img=True, wireframe=False)

    
