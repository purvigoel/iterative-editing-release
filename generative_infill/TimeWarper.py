import numpy as np
from generative_infill.change_time.timewarp import TimeWarp


class TimeWarpWrapper:
    def __init__(self, model, max_frames, max_frames2=None):
        self.model = model
        self.time_warp = TimeWarp(max_frames, max_frames2)

    def forward(self, input_motions, speed, start_time, end_time, dump_frames=False):
        interpolated_version = input_motions.clone()
        if speed == "pause":
            interpolated_version = self.time_warp.pause_ctrl(interpolated_version, start_time, end_time)
        elif speed == "fast":
            interpolated_version = self.time_warp.speed_up_ctrl(interpolated_version, start_time, end_time)
        elif speed == "slow":
            interpolated_version = self.time_warp.slow_down_ctrl(interpolated_version, start_time, end_time)
        else:
            print("couldnt recognize speed")
            assert(1==0)
        
        return interpolated_version

    def forward_dump(self, input_motions, speed, start_time, end_time, dump_frames=True):
        interpolated_version = input_motions.clone()
        if speed == "pause":
            interpolated_version = self.time_warp.pause_ctrl(interpolated_version, start_time, end_time, dump_frames)
        elif speed == "fast":
            interpolated_version = self.time_warp.speed_up_ctrl(interpolated_version, start_time, end_time, dump_frames)
        elif speed == "slow":
            interpolated_version = self.time_warp.slow_down_ctrl(interpolated_version, start_time, end_time, dump_frames)
        else:
            print("couldnt recognize speed")
            assert(1==0)

        return interpolated_version

    def forward_ctrl(self, input_motions, old_frame, new_frame, ctrl_pts=None, ctrl=None):
        interpolated_version = input_motions.clone()
        interpolated_version = self.time_warp.direct_ctrl(interpolated_version, old_frame, new_frame, ctrl_pts, ctrl)
        return interpolated_version
