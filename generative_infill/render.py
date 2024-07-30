import pyvista as pv
import os
import imageio
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings("ignore")

pv.start_xvfb()

def adjust_camera(plotter):
    camera = plotter.camera
    current_position = camera.position
    focal_point = camera.focal_point
    #view_up = camera.view_up

    # Calculate a new camera position that is further back
    # This is done by increasing the distance between the camera position and the focal point
    distance_factor = 2.0  # Adjust this factor to move the camera back more or less
    new_position = (
    current_position[0] + (current_position[0] - focal_point[0]) * (distance_factor - 1),
    current_position[1] + (current_position[1] - focal_point[1]) * (distance_factor - 1),
    current_position[2] + (current_position[2] - focal_point[2]) * (distance_factor - 1)
    )
    # Update the camera position
    camera.position = new_position
    camera.position = [camera.position[0], camera.position[1] , -0.5] # 0.5 0.5 - 0.5]
    return camera.position, camera.focal_point

def add_floor(character_mesh, ground_z=-1.1509):
    ground_z = ground_z
    print(character_mesh.bounds)
    floor_bounds = [-5, 5, -5, 5, ground_z - 0.5, ground_z]
    print(floor_bounds)
    # Create the floor (a flat box)
    floor = pv.Box(bounds=floor_bounds)
    return floor

def add_light():
    light = pv.Light()
    light.position = (10, 5, 15)  # Adjust the position as needed
    light.intensity = 0.1
    return light

def floor_loc(folder, T, image_dir):
    mesh_files = []
    for i in range(T):
        mesh_files.append( folder + "/" + str(i).zfill(3) + ".obj")

    # Render each mesh to an image
    for i, file in enumerate(tqdm(mesh_files)):
        start = time.time()
        mesh = pv.read(file)
        mesh.rotate_x(90)



        floor = add_floor(mesh, y)

def render_images(folder, T, image_dir="temp_images"):
    os.makedirs(image_dir, exist_ok=True)

    mesh_files = []
    for i in range(T):
        mesh_files.append( folder + "/" + str(i).zfill(3) + ".obj")
    
    mesh = pv.read( mesh_files[0] )
    mesh.rotate_x(90)
    y = mesh.bounds[4]
    print(y, mesh.bounds)
    render_start = time.time()
    # Render each mesh to an image
    for i, file in enumerate(tqdm(mesh_files)):
        start = time.time()
        mesh = pv.read(file)
        print("loading", time.time() - start)
        mesh.rotate_x(90)
        mesh.rotate_z(180 + 25 - 180 + 180 - 45 )

        plotter = pv.Plotter(off_screen=True, window_size=[400,400])
        plotter.background_color = "lightgray"
        light = add_light()
        plotter.add_light(light)

        plotter.add_mesh(mesh, color='cornflowerblue') #'cornflowerblue') #'darkseagreen') #'cornflowerblue')
        
        
        if i == 0:
            position, focal_point = adjust_camera(plotter)
        
        floor = add_floor(mesh, y)
        plotter.add_mesh(floor, color='gray')
        
        plotter.camera.position = position
        plotter.camera.focal_point = focal_point
        plotter.enable_shadows()

        start = time.time()
        plotter.show(screenshot=os.path.join(image_dir, f'image_{i:03}.png'), window_size=[400,400])
        print("render", time.time() - start)
    print("total", time.time() - render_start)

def images_to_vid(image_dir="temp_images", output_video="output_video.mp4"):
    # Get list of all image files
    image_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('.png')]

    # Create a video writer
    writer = imageio.get_writer(output_video, fps=24)  # Adjust FPS as needed

    # Add images to video
    for img_file in image_files:
        img = imageio.imread(img_file)
        writer.append_data(img)
    # Close the writer to finalize the video
    writer.close()

def render_and_save(folder, T=60, image_dir="/raid/pgoel2/bio-pose/temp_images", output_video="output_video.mp4"):
    render_images(folder, T, image_dir)
    images_to_vid(image_dir, folder + "/" + output_video)

render_and_save("/raid/pgoel2/bio-pose/user_study_siggraph/out/", T=60)
