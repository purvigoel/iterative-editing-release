import torch
import numpy
from meo_constants import meoConstant

class TrajectoryWrapper:
    def __init__(self, model, diffusion, data_template, dist_util, max_frames=meoConstant.max_frames):
        self.model = model
        self.diffusion = diffusion
        self.max_frames = max_frames
        print(self.max_frames)
        self.preserve_frames = []
        self.preserve_key = []
        self.dist_util = dist_util
        self.data_template = data_template

    def prepare_sample(self, original_motion, keyframed_motion, frames, window, hi_window, i):
        original = original_motion.clone()
        keyframe = keyframed_motion.clone()
        
        for frame in frames:
            lo = max(0, frame-window)
            if not hi_window:
                hi_window = window

            hi = min(self.max_frames, frame + hi_window)
            original[i, :, :, lo:frame] = 0
            original[i, :, :, frame + 1:hi] = 0
        
        for frame in frames:
            original[i, :, :, frame] = keyframe[i, :, :, frame]

        if isinstance(self.preserve_frames, list):
            for p in self.preserve_frames:
                old_frame = p
                if old_frame not in frames:
                    old_keyframe = self.preserve_key[i, ..., old_frame]
                    original[i, ..., old_frame] = old_keyframe

        original[i, 6:24*6, :, :] = 0
        original[i, 25*6:, :, :] = 0

        #self.preserve_frames.extend(frames)
        return original

    def forward(self, original_motion, keyframed_motion, frames, lo_window=15, hi_window=15, i=0):
        iterator = iter(self.data_template)
        input_motions, model_kwargs = next(iterator)

        input_motions = original_motion.clone()

        model_kwargs['y']['text'] = [''] * original_motion.shape[0]
        guidance_param = 0.
        model_kwargs['y']['scale'] = torch.ones(original_motion.shape[0], device=self.dist_util.dev()) * guidance_param

        original = self.prepare_sample(original_motion, keyframed_motion, frames, lo_window, hi_window, i)
        model_kwargs["y"]["keyframe"] = original.squeeze(2).permute(0, 2, 1)

        input = torch.zeros(input_motions.shape).to(self.dist_util.dev())
    
        sample = self.model(input, torch.zeros(input_motions.shape[0]).long().to(self.dist_util.dev()), **model_kwargs)
        sample = sample.detach()

        sample[i, 6:24*6, :, :] = 0
        sample[i, 25*6:, :, :] = 0

        return sample
