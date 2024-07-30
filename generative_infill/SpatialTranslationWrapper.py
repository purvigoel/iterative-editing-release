import numpy as np
import torch
import pose_mask
import time
from generative_infill.ik_solver2 import IKSolver

class UnconditionalMDMWrapper:
    def __init__(self, model, diffusion):
        self.model = model
        self.diffusion = diffusion

class SpatialTranslationWrapper:
    def __init__(self, ikGuider, data_template, dist_util):
        self.data_template = data_template
        self.dist_util = dist_util
        self.ikGuider = ikGuider
        self.ik_solver = IKSolver()

    def forward(self, sample, edit_limb, edit_frame, magx, magy, magz, contact):
        if edit_limb == "right_foot":
            sample = self.ik_solver.to_target(sample, self.ikGuider.model, edit_frame, torch.tensor([magx, magy, magz]), side="right_foot")
            return sample
        elif edit_limb == "left_foot":
            sample = self.ik_solver.to_target(sample, self.ikGuider.model, edit_frame, torch.tensor([magx, magy, magz]), side="left_foot")
            return sample

        mags = np.zeros( (sample.shape[0], 3 ))
        limbs = []
        framenums = []

        counter = 0
        for i in range(0, sample.shape[0]): 
            mags[counter,0] = magx 
            mags[counter,1] = magy
            mags[counter,2] = magz
            limbs.append(edit_limb)
            framenums.append(edit_frame)
            counter += 1
        mags = torch.tensor(mags).to(self.dist_util.dev())

        iterator = iter(self.data_template)
        input_motions, model_kwargs = next(iterator)

        input_motions_motionA = sample.clone() 

        input_motions = input_motions_motionA.float()

        input_motions = input_motions.to(self.dist_util.dev())

        clone_inputs = input_motions.clone()

        texts = [''] * sample.shape[0]
        model_kwargs['y']['text'] = texts
        guidance_param = 0.


        N = sample.shape[0]
        T = sample.shape[-1]
    
        no_root = input_motions.clone()

        j_dic = self.ikGuider.model.forward_kinematics(input_motions, None)

        pose_joints = j_dic['kp_45_joints'][:, :24].reshape(N, T, 24, 3 )
        pose_joints = pose_joints + torch.tensor([0, 0.2256, 0]).repeat(N, T, 1).unsqueeze(-2).to(input_motions.device)

        smpl_joints = pose_joints.view(N, T, 24 * 3, 1)
        smpl_joints = smpl_joints.permute(0, 2, 3, 1)
        input_motions[:, 164 : 236, :, :] = smpl_joints.clone()

        model_kwargs['y']['inpainted_motion'] = input_motions
        print(input_motions.shape)

        model_kwargs['y']['inpainting_mask'] = torch.tensor(pose_mask.SMPL_IKSOLVER_MASK, dtype=torch.bool,
                                                            device=input_motions.device)  # True is lower body data
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])
        model_kwargs['y']['inpainting_mask'][...] = 1

        model_kwargs['y']['ratio'] = 1
        model_kwargs['y']['blend'] = 1.0
        model_kwargs['y']['ik'] = True

        model_kwargs['y']['framenum'] = framenums
        model_kwargs['y']['mag'] = mags
        model_kwargs['y']['limb'] = limbs

        if not isinstance(contact, type(None)) and len(contact) > 0:
            model_kwargs['y']['contact'] = contact

        start = time.time()
        
        
        N = sample.shape[0]
        T = sample.shape[-1]
        video_features = torch.zeros(N, T, 2048)
        model_kwargs["y"]["features"] = torch.zeros(N, T, 2048)

        print(f'### Start sampling [repetitions]')
        model_kwargs['y']['scale'] = torch.ones(sample.shape[0], device=self.dist_util.dev()) * guidance_param

        with torch.no_grad():
            sample_fn = self.ikGuider.diffusion.p_sample_loop
            sample = sample_fn(
            self.ikGuider.model,
            (sample.shape[0], self.ikGuider.model.njoints, self.ikGuider.model.nfeats, sample.shape[-1]),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            grad_model=self.ikGuider.model
            )

        framenums = model_kwargs["y"]["framenum"]
        s = []
        for f in range(len(framenums)):
            framenum = framenums[f]
            s.append(sample[f, ..., framenum].unsqueeze(-1))
        sample = torch.stack(s)
            
        sample = sample.detach()
        condition = clone_inputs
        old_condition = condition.clone() 
        
        if(edit_limb == "right_hand"):
            condition[:, pose_mask.right_arm_angles, :, edit_frame] = sample[:, pose_mask.right_arm_angles].squeeze(-1) 
            sample = condition[..., edit_frame].unsqueeze(-1).clone()
        elif(edit_limb == "left_hand"):
            condition[:, pose_mask.left_arm_angles_kc, :, edit_frame] = sample[:, pose_mask.left_arm_angles_kc].squeeze(-1)
            sample = condition[..., edit_frame].unsqueeze(-1).clone()
        old_condition[:, :24*6,..., edit_frame] = sample[:,:24*6, ..., 0]
        sample = old_condition
        return sample
