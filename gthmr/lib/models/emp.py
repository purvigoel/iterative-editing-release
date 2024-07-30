from mdm.model.mdm import *
import mdm.utils.rotation_conversions as geometry
import roma
import os.path as osp
# HMR related
from VIBE.lib.core.config import VIBE_DATA_DIR
from VIBE.lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS, SMPL_TO_J14
from dpm_solver import DPM_Solver, model_wrapper, NoiseScheduleVP
from misc_utils import to_np, to_tensor
from VIBE.lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from VIBE.lib.models.spin import projection
from gthmr.lib.utils.data_utils import split_rot6d_extra, combine_rot6d_extra, rotate_motion_by_rotmat


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PitchCorrector0(nn.Module):
    """
    First model to predict pitch correction, i.e., from Sideline view to Camera View. 
    This uses only the processed video features.  
    """

    def __init__(self, input_feat_dim, rot_type='aa'):
        super(PitchCorrector0, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.mlp = MLP(self.input_feat_dim, 100, 6)

    def forward(self, feat):
        """
        Input
            feat -- (N, T, D)
        """
        N, T, D = feat.shape
        out = self.mlp(feat.reshape(-1, D)).reshape(N, T, 6).sum(1)
        return out  # (N, 6)


class HMRHead(nn.Module):
    """
    First model to predict hmr stuff: betas and cam
    """

    def __init__(self, input_feat_dim):
        super(HMRHead, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.beta_mlp = MLP(self.input_feat_dim, 100, 10)
        self.cam_mlp = MLP(self.input_feat_dim, 100, 3)

    def forward(self, feat):
        """
        Input
            feat -- (N, T, D)
        """
        N, T, D = feat.shape
        beta = self.beta_mlp(feat.reshape(-1, D)).reshape(N, T, 10).sum(1)
        cam = self.cam_mlp(feat.reshape(-1, D)).reshape(N, T, 3).sum(1)
        return beta, cam  # (N, 10), (N, 3)


class EMP(MDM):

    def __init__(self,
                 modeltype,
                 njoints,
                 nfeats,
                 num_actions,
                 translation,
                 pose_rep,
                 glob,
                 glob_rot,
                 latent_dim=256,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.1,
                 ablation=None,
                 activation="gelu",
                 legacy=False,
                 data_rep='rot6d',
                 dataset='amass',
                 clip_dim=512,
                 arch='trans_enc',
                 emb_trans_dec=False,
                 clip_version=None,
                 video_dim=2048,
                 video_arch='linear',
                 **kargs):
        super().__init__(modeltype,
                         njoints,
                         nfeats,
                         num_actions,
                         translation,
                         pose_rep,
                         glob,
                         glob_rot,
                         latent_dim=latent_dim,
                         ff_size=ff_size,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         dropout=dropout,
                         ablation=ablation,
                         activation=activation,
                         legacy=legacy,
                         data_rep=data_rep,
                         dataset=dataset,
                         clip_dim=clip_dim,
                         arch=arch,
                         emb_trans_dec=emb_trans_dec,
                         clip_version=clip_version,
                         video_dim=video_dim,
                         video_arch=video_arch,
                         **kargs)

        # Module for Cam-View Pitch Correction
        self.pitch_corrector = PitchCorrector0(self.latent_dim)

        # Module for HMR output
        self.hmr_module = HMRHead(self.latent_dim)

        # HMR Modules: Regressor, SMPL
        #self.J_regressor = torch.from_numpy(
        #    np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()
        self.smpl = SMPL("./body_models/smpl/", batch_size=64, create_transl=False)

        # DPM-Solver
        self.dpm_solver_args = {
            'algorithm_type': "dpmsolver++",
            'method': 'multistep',
            'order': 3,
            'steps': 20,
        }

    def model_fn(self, x, t, model_kwargs):
        model = self
        diffusion = self.diffusion
        return diffusion.p_mean_variance(model,
                                         x,
                                         t,
                                         model_kwargs=model_kwargs,
                                         clip_denoised=False)['pred_xstart']

    def inference(self,
                  video_features,
                  use_dpm_solver=False,
                  steps=None,
                  order=None,
                  denoise_to_zero=True,
                  skip_timesteps=0,
                  overwrite_kwargs = None):
        """
        Infer the 3D motion in the Sideline view given video features.  
        This runs sampling of the video-conditioned MDM. 
        Input
            video_features -- Tensor of (N, T, D) where D is usually 2048
            use_dmp_solver (bool). If True, dpm solver, else do standard 1000-step 
            sampling. DPM-solver is much faster, though *may* require tuning for new
            datasets.
            steps: parameter for DPM solver.  
            denoise_to_zero -- dpm solver flag
            steps -- dpm solver value
        """
        diffusion = self.diffusion
        device = video_features.device
        N, T, _ = video_features.shape
        noise_shape = (N, self.njoints, self.nfeats, T)
        model_kwargs = {'y': {'features': video_features}}
        if overwrite_kwargs:
            model_kwargs = overwrite_kwargs

        if not use_dpm_solver:
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(
                self,
                noise_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=skip_timesteps,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        else:
            # Init DPM Solver
            noise_schedule = NoiseScheduleVP(
                betas=to_tensor(self.diffusion.betas))
            wrapped_model_fn = model_wrapper(
                lambda x, t: self.model_fn(x, t, model_kwargs),
                noise_schedule,
                model_type='x_start')

            self.dpm_solver = DPM_Solver(
                wrapped_model_fn,
                noise_schedule,
                algorithm_type=self.dpm_solver_args['algorithm_type'])

            noise = torch.randn(noise_shape).to(device)

            steps = self.dpm_solver_args['steps'] if steps is None else steps
            order = self.dpm_solver_args['order'] if order is None else order
            sample = self.dpm_solver.sample(
                x=noise,
                method=self.dpm_solver_args['method'],
                order=order,
                steps=steps,
                skip_type="logSNR",
                denoise_to_zero=denoise_to_zero,
                no_grad=False
                # optional: skip_type='time_uniform' or 'logSNR' or 'time_quadratic',
            )
        return sample

    def run_pitch_correction(self, video_features, slv_motion):
        """
        Apply pitch correction to a Sideline View motion based on `video_features`.

        Input
            slv_motion -- (N, 25, 6, T) motion where the last joint is XYZ000.
        Return
            cv_motion -- motion of the same shape, but rotated.
        """
        raise  # not used anynmore
        # 1. Embed video features by re-using MDM conditioning.
        features = video_features.cuda()
        shape = features.shape
        video_emb = self.embed_video(features)  # (N,T,D)

        # 2. Forward pass with pitch corrector module
        pitch = self.pitch_corrector(video_emb)  # (N, 6)

        # 3. Apply `pitch` to `slv_motion`
        rotmat = geometry.rotation_6d_to_matrix(pitch)  # (N, 3, 3)
        cv_motion = rotate_motion_by_rotmat(slv_motion, rotmat)
        return cv_motion

    def forward_kinematics(self, cv_motion, betas):
        """
        Run forward kinematics with SMPL to get 3D joints.

        Input
            cv_motion -- either (N, 25, 6, T) or (N, 154, 6, T) or (N, 164, 6, T) 
                motion in Camera view.
            betas -- (N, 10)
        Return 
            a dictionary of j3d in the shape of (N * T, K, 3) where K varies.
        """
        N, J, D, T = cv_motion.shape
        cv_motion, _ = split_rot6d_extra(
            cv_motion)  # get the 6d motion if not in rot6d format

        # Re-format motion to SMPL input format (i.e., aa)
        cv_motion = cv_motion.permute(0, 3, 1, 2).reshape(N * T, 25, 6)
        pred_rotmat = geometry.rotation_6d_to_matrix(
            cv_motion[:, :24])  # (N * T, 24, 3, 3)
        pred_trans = cv_motion[:, [-1], :3]  # (N * T, 1, 3)
        # Expand betas
        if betas is not None:
            # case 1: (N,10). Each seq has 1 beta. Copy it in time dim to (N,T,10), then shape (N*T,10)
            if betas.ndim == 2:
                betas = betas.unsqueeze(1).repeat(1, T, 1).reshape(N * T, -1)
            # case 2: shape is (N,T,10). Shape to (N*T,10)
            elif betas.ndim == 3:
                betas = betas.reshape(N * T, -1)
            else:
                raise
        else:
            betas = torch.zeros((N * T, 10)).to(cv_motion.device)
        global_orient = pred_rotmat[:, [0]]
        #global_orient = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(global_orient.shape[0], 1, 1, 1).to(global_orient.device)

        # SMPL forward kinematics
        pred_output = self.smpl(betas=betas,
                                body_pose=pred_rotmat[:, 1:],
                                global_orient=global_orient,
                                pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints  # SPIN joints
        kp_45_joints = pred_output.smpl_joints  # SMPL+EXTRA
        # Extract the 14 Common joints.
        #J_regressor = self.J_regressor
        #J_regressor_batch = J_regressor[None, :].expand(
        #    pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        #kp_c14_joints = torch.matmul(J_regressor_batch,
        #                             pred_vertices)[:, H36M_TO_J14, :]
        kp_c14_joints = None
        return {
            "pred_vertices": pred_vertices,
            "pred_joints": pred_joints,  # 49 SPIN joints
            "kp_45_joints":
            kp_45_joints,  # 45 SMPL+EXTRA joints (first 22 match SMPL)
            'kp_c14_joints': kp_c14_joints,  # 14 Common joints
            'pred_trans': pred_trans,
            'pred_output': pred_output,
            'pred_rotmat': pred_rotmat,
            "faces": self.smpl.faces
        }

    def inference_cam_view(self,
                           video_features,
                           use_dpm_solver=False,
                           denoise_to_zero=True,
                           steps=None):
        """
        Infer the 3D motion in the Camera view given video features.
        This is used for HMR prediction.
        It first runs `inference`, then performs `pitch_correction`. 

        Input
            video_features -- Tensor of (N, T, D) where D is usually 2048
        """
        # Runs `inference`
        with torch.no_grad():
            slv_motion = self.inference(video_features,
                                        use_dpm_solver=use_dpm_solver,
                                        denoise_to_zero=denoise_to_zero,
                                        steps=steps)

        # Pitch correction
        # cv_motion = self.run_pitch_correction(video_features, slv_motion)
        cv_motion = slv_motion

        return cv_motion

    def hmr_forward(self, video_features, cv_motion, mean_beta=True):
        """ Get predicted shape and camera. Depends on the data type. If 
        """
        if 'shape' in self.data_rep:
            N, J, D, T = cv_motion.shape
            assert D == 1
            rot6d, extra = split_rot6d_extra(cv_motion)
            beta = extra[:, 4:14]  # (N,10,1,T)

            if mean_beta:
                beta = beta.mean(-1).squeeze(-1)  # (N,10)
            else:
                beta = beta.squeeze(2).permute(0, 2, 1)  # (N,T,10)
            cam = torch.zeros((N, 3),
                              dtype=beta.dtype,
                              device=beta.device,
                              requires_grad=True)  # (N,3)

        else:
            # 1. Embed video features by re-using MDM conditioning.
            features = video_features.cuda()
            shape = features.shape
            video_emb = self.embed_video(features)

            # 2. Forward pass with HMR module
            beta, cam = self.hmr_module(video_emb)  # (N, 10), (N, 3)

        return beta, cam

    def inference_hmr(self,
                      video_features,
                      use_dpm_solver=False,
                      denoise_to_zero=True,
                      steps=None,
                      J_regressor=None):
        """
        Infer full HMR output.

        Input
            video_features -- Tensor of (N, T, D) where D is usually 2048
        """
        N, T, _ = video_features.shape
        B = N * T

        # MDM pass
        cv_motion = self.inference_cam_view(video_features,
                                            use_dpm_solver=use_dpm_solver,
                                            steps=steps,
                                            denoise_to_zero=denoise_to_zero)

        # HMR pass (betas, cameras)
        pred_shape, pred_cam = self.hmr_forward(video_features, cv_motion)

        # FK
        fk_dict = self.forward_kinematics(cv_motion, pred_shape)

        #######################################################################
        #
        # Note: Everything will have a batch size of B = N * T
        #
        #######################################################################
        # Repeat int he time dimension
        pred_shape = pred_shape.unsqueeze(1).repeat(1, T, 1).reshape(B, -1)
        pred_cam = pred_cam.unsqueeze(1).repeat(1, T, 1).reshape(B, -1)

        # Format into HMR output
        pred_vertices = fk_dict['pred_vertices']
        pred_joints = fk_dict['pred_joints']
        root_traj = fk_dict['pred_trans']
        pred_output = fk_dict['pred_output']
        pred_rotmat = fk_dict['pred_rotmat']

        smpl_joints_w_trans = pred_output.smpl_joints + root_traj.reshape(
            -1, 1, 3)

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(
                pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3,
                                                                 3)).reshape(
                                                                     -1, 72)
        output = [{
            'theta': torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'smpl_output': pred_output,
            'smpl_joints_w_trans': smpl_joints_w_trans,
            'rotmat': pred_rotmat,
            'root_traj': root_traj,
        }]

        return output
