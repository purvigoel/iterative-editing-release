from generative_infill.reloader import *
from meo_constants import meoConstant

class InfillWrapper:
    def __init__(self, model, diffusion, data_template, dist_util, max_frames=meoConstant.max_frames):
        self.model = model 
        self.diffusion = diffusion
        self.preserve_frames = []
        self.preserve_key = []
        self.data_template = data_template
        self.max_frames = max_frames
        self.dist_util = dist_util

    def prepare_sample(self, input_motions, keyframe, trajectory_sample, window, hi_window, frames, i, is_root_edit):

        original = input_motions.clone()
        keyframe = reload(self.model, keyframe, include_fc = False)
        keyframe[:, 150:154, :, frames ] = 0

        trajectory = trajectory_sample.clone()
        keyframe_mask = torch.ones(original.shape).to(original.device).to(torch.bool)

        for frame in frames:
            lo = max(0, frame - window)
            if not hi_window:
                hi_window = window
            hi = min(self.max_frames, frame + hi_window)
            original[i, :, :, lo:frame] = trajectory[i, :, :, lo:frame]
            original[i, :, :, frame + 1:hi] = trajectory[i, :, :, frame+1:hi]
            keyframe_mask[i, 6:24*6, :, lo:frame] = 0
            keyframe_mask[i, 25*6:, :, lo:frame] = 0
            keyframe_mask[i, 6:24*6, :, frame+1:hi] = 0
            keyframe_mask[i, 25*6:, :, frame+1:hi] = 0

        for frame in frames:
            print(i, frame)
            original[i, :, :, frame] = keyframe[i, :, :, frame]
            keyframe_mask[i, :, :, frame] = 1        

        if isinstance(self.preserve_frames, list): 
            for p in self.preserve_frames:
                old_frame = p
                if old_frame != frame:
                    old_keyframe = self.preserve_key[i, ..., old_frame]
                    original[i, ..., old_frame] = old_keyframe
                    keyframe_mask[i, ..., old_frame] = 1

        to_infill = input_motions.clone()

        if not is_root_edit:
            og_motion = to_infill.clone()
            for frame in frames:
                keyf = keyframe[:, :, :, frame]

                delta = keyf.clone() - og_motion[:,:,:,frame].clone()
                lo = max(0, frame - window)
                if not hi_window:
                    hi_window = window
                hi = min(self.max_frames, frame + hi_window)

                for fr in range(lo, frame):
                    a = (fr - lo) / (frame - lo)
                    og_motion[:, :, :, fr] = og_motion[:, :, :, fr] + (a) * (delta)
                for fr in range(frame, hi):
                    a = (fr - frame) / (hi - frame)
                    og_motion[:, :, :, fr] = og_motion[:, :, :, fr] + (1 - a) * (delta)
            
            to_infill[:, :24*6, : , :] = og_motion[:, :24*6, : , :].clone()
        
        keyframed_version = original.clone()
        interpolated_version = to_infill.clone()

    
        return keyframed_version, interpolated_version, keyframe_mask

    def forward(self, original_motion, keyframe, trajectory_sample, lo_window, hi_window, frames, i = 0, is_root_edit=False ):
        iterator = iter(self.data_template)
        input_motions, model_kwargs = next(iterator)
        
        model_kwargs['y']['text'] = [''] * input_motions.shape[0]
        guidance_param = 0.

        original_motion = original_motion.to(self.dist_util.dev())
        keyframe = keyframe.to(self.dist_util.dev())
        trajectory_sample = trajectory_sample.to(self.dist_util.dev())

        keyframed_version, interpolated_version, keyframe_mask = self.prepare_sample(original_motion, keyframe, trajectory_sample, lo_window, hi_window, frames, i, is_root_edit)
        keyframed_version = keyframed_version.to(self.dist_util.dev())
        interpolated_version = interpolated_version.to(self.dist_util.dev())
        keyframe_mask = keyframe_mask.to(self.dist_util.dev())

        model_kwargs["y"]["keyframe"] = keyframed_version.squeeze(2).permute(0, 2, 1)
        noise = torch.randn(keyframed_version.shape).to(keyframed_version.device)
        noise = torch.where(keyframe_mask, keyframed_version, noise)
       
        if not is_root_edit:
            model_kwargs["y"]["keyframe_mask"] = keyframe_mask
            clonemask = keyframe_mask.clone()
            clonemask = torch.where(keyframe_mask, 1.0, 0.1)
            model_kwargs["y"]["inpainting_mask"] = clonemask #keyframe_mask.clone() #torch.ones(original.shape).to(original.device).to(torch.bool)
        
            model_kwargs["y"]["inpainted_motion"] = interpolated_version.clone()
            model_kwargs["y"]["upsample_infill"] = True
            model_kwargs["y"]["ratio"] = 999

        original = original_motion.clone()
        model_kwargs["y"]["noise_mask"] = keyframe_mask.clone()
        model_kwargs["y"]["original"] = keyframed_version.clone()

        N = keyframed_version.shape[0]
        T = keyframed_version.shape[-1]

        video_features = torch.zeros(N, T, 2048)
        model_kwargs["y"]["features"] = torch.zeros(N, T, 2048)
        with torch.no_grad():
            sample_fn = self.diffusion.p_sample_loop
            sample = sample_fn(
                self.model,
                (N, self.model.njoints, self.model.nfeats, self.max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=noise,
                const_noise=False,
                const_t_noise=False,
                grad_model=self.model
            )
            sample = sample.detach()
        sample = reload(self.model, sample)
        return sample, interpolated_version

