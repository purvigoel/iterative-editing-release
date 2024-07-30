import numpy as np
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator
import math
from scipy.optimize import brentq
import pose_mask

def create_piecewise_polynomial(control_points):
    x = [point[0] for point in control_points]
    y = [point[1] for point in control_points]
    spline = PchipInterpolator(x, y) #CubicSpline(x, y)
    return spline

def sample_spline(spline, control_points, itr=60):
    x_min = min(point[0] for point in control_points)
    x_max = max(point[0] for point in control_points)
    x_vals = np.linspace(x_min, x_max, itr)
    y_vals = spline(x_vals)
    return y_vals

def create_time_warp(control_point_set):
    spline = create_piecewise_polynomial(control_points)
    return spline

class TimeWarp:
    def __init__(self, max_time, maxtime2=None):
        self.min = 0
        self.max_time = max_time
        if not maxtime2:
            self.max_time2 = self.max_time
        else:
            self.max_time2 = maxtime2
        self.control_points = [ (0, 0), (self.max_time, self.max_time2) ]
        self.curr_spline = None
        self.update_warp()
        self.mapping = None

    def update_warp_multiple(self, points):
        for pt in points:
            if pt[0] >= self.max_time:
                self.control_points[1] = pt
            else:
                self.control_points.append(pt)
        
        #self.control_points.extend(points)
        sorted_list = sorted(self.control_points)
        #print(sorted_list)
        self.control_points = sorted_list
        self.curr_spline = create_piecewise_polynomial(self.control_points)
        self.mapping = sample_spline(self.curr_spline, self.control_points, self.max_time)

    def update_warp(self, control_point=None):
        if control_point:
            is_valid = True
            is_handled = False
            for i in range(len(self.control_points)):
                if control_point[0] == self.control_points[i][0] and control_point[1] == self.control_points[i][1]:
                    is_valid = False
                    break
                if control_point[0]== self.control_points[i][0]:
                    is_handled = True
                    is_valid = False

                    self.control_points[i] = (control_point[0], control_point[1])
                    sorted_list = sorted(self.control_points)
                    self.control_points = sorted_list
                    break

            if is_valid:    
                self.control_points.append(control_point)
                sorted_list = sorted(self.control_points)
                self.control_points = sorted_list
        self.curr_spline = create_piecewise_polynomial(self.control_points)
        self.mapping = sample_spline(self.curr_spline, self.control_points, self.max_time)

    def speed_up(self, motion, frame):
        curr_frame_warp_val = self.curr_spline(frame)
        curr_frame_warp_val = min(curr_frame_warp_val + 10, self.max_time)

        self.update_warp( (frame, curr_frame_warp_val) )
        return self.bake(motion)

    def speed_up_ctrl(self, motion, start_frame, end_frame, dump_frames=False):
        if start_frame == end_frame:
            frame = start_frame
            curr_frame_warp_val = self.curr_spline(frame)
            curr_frame_warp_val = min(curr_frame_warp_val + 10, self.max_time)
            self.update_warp( (frame, curr_frame_warp_val) )
        else:
            start_time_warp_val = self.curr_spline(start_frame)
            curr_frame_warp_val = self.curr_spline(end_frame)
            curr_frame_warp_val = min(curr_frame_warp_val + 10, self.max_time)

            self.update_warp( (start_frame, start_time_warp_val))
            self.update_warp( (end_frame, curr_frame_warp_val) )
        return self.bake(motion, dump_frames)

    def slow_down(self, motion, frame):
        curr_frame_warp_val = self.curr_spline(frame )
        curr_frame_warp_val = max(curr_frame_warp_val - 10, 0)

        self.update_warp( (frame, curr_frame_warp_val) )
        return self.bake(motion)

    def slow_down_ctrl(self, motion, start_frame, end_frame, dump_frames=False):
        if start_frame == end_frame:
            frame = start_frame
            curr_frame_warp_val = self.curr_spline(frame)
            curr_frame_warp_val = max(curr_frame_warp_val - 10, 0)
            self.update_warp( (frame, curr_frame_warp_val) )
        else:
            start_time_warp_val = self.curr_spline(start_frame)
            curr_frame_warp_val = self.curr_spline(end_frame)
            curr_frame_warp_val = max(curr_frame_warp_val - 10, 0)
            self.update_warp( (start_frame, start_time_warp_val))
            self.update_warp( (end_frame, curr_frame_warp_val) )
        return self.bake(motion, dump_frames)

    def direct_ctrl(self, motion, old_frame, new_frame, ctrl_pts=None, ctrl=False):
        if ctrl:
            self.update_warp_multiple(ctrl_pts)
        else:
            self.update_warp( (old_frame, new_frame))
        return self.bake(motion)

    def pause(self, motion, frame):
        curr_frame_warp_val = self.curr_spline(frame)
        lo_frame = max(0, frame - 2)
        hi_frame = min(self.max_time, frame + 2)

        self.update_warp( (lo_frame, curr_frame_warp_val))
        self.update_warp( (hi_frame, curr_frame_warp_val))
        return self.bake(motion)

    def pause_ctrl(self, motion, start_time, end_time, dump_frames=False):
        if start_time == end_time:
            frame = start_time
            curr_frame_warp_val = self.curr_spline(frame)
            lo_frame = max(0, frame - 1)
            hi_frame = min(self.max_time, frame + 1)

            self.update_warp( (lo_frame, curr_frame_warp_val))
            self.update_warp( (hi_frame, curr_frame_warp_val))
        else:
            start_time_warp_val = self.curr_spline(start_time)
            hi_frame = end_time
            lo_frame = start_time
            self.update_warp( (lo_frame, start_time_warp_val)) 
            self.update_warp( (hi_frame, start_time_warp_val))
        return self.bake(motion, dump_frames)

    def find_x_for_y(self, y_target, x_range=(0,60)):
        # Define the function to find roots of
        func = lambda x: self.curr_spline(x) - y_target

        # Find the root (x value) within the specified range
        x_value = brentq(func, x_range[0], x_range[1])

        return x_value

    def graph(self):
        vals = []
        for i in range(self.max_time):
            x_val = self.curr_spline(i)
            vals.append(x_val)
        return vals

    def get_new_inds(self):
        frames = []
        for i in range(self.max_time2):
            frames.append( self.find_x_for_y(i))
            #frames.append(self.curr_spline(i).item())
        new_frames = []
        for i in range(self.max_time2):
            if i == 0:
                new_frames.append(frames[i])
            else:
                new_frames.append(frames[i] - frames[i - 1])
        return new_frames

    def bake(self, motion, dump_frames=False):
        old_motion = motion.clone()
        new_motion = np.zeros( (motion.shape[0], motion.shape[1], motion.shape[2], self.max_time2)) #clone()
        frames = []


        for i in range(self.max_time2):
            x_val = self.curr_spline(i)
            #x_val = self.find_x_for_y(self.curr_spline, i, [0, self.max_time])
            x_val_floor = math.floor(x_val)
            x_val_ceil = x_val_floor + 1

            x_val_floor_mult = x_val - x_val_floor
            x_val_ceil_mult = x_val_ceil - x_val

            x_val_floor = min(max(0, x_val_floor), self.max_time - 1)
            x_val_ceil = min(max(0, x_val_ceil), self.max_time - 1)
            #print(i, x_val)
            frames.append(x_val)
            new_frame = motion[:, :, :, 0].clone()
            new_frame[ :, :, :]  = motion[:, :, :, max(0, x_val_floor)] * (1.0 - x_val_floor_mult) + motion[:, :, :, min(x_val_ceil, self.max_time - 1) ] * (1.0 - x_val_ceil_mult)
            new_motion[..., i] = new_frame

        if dump_frames:
            return new_motion, frames
        return new_motion
        
    def bake_differentiable(self, motion, frs, normalized = False):
         
        grid = torch.zeros(motion.shape[0], motion.shape[2], motion.shape[3], 2).to(motion.device)
        
        if not normalized:
            frs =  (frs / 30) - 1

        if not normalized:
            for i in range(1, frs.shape[1]):
                grid[:, :, i, 1] = -1
                grid[:, :, i, 0] = frs[:, i].unsqueeze(1)
        else:
            
            for i in range(0, frs.shape[1]):
                grid[:, :, i, 1] = -1
                if i == 0:
                    grid[:, :, i, 0] = frs[:, i].unsqueeze(1)
                else:
                    grid[:, :, i, 0] = frs[:, i].unsqueeze(1) + grid[:, :, i - 1, 0]

            grid[:, :, :, 0] = grid[:, :, :, 0] / 30 - 1
        
        new_motion = torch.nn.functional.grid_sample( motion, grid) 
        return new_motion


