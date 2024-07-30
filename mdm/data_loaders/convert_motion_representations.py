import numpy as np
import pandas as pd
import os
import ipdb
import torch
import tqdm

from gthmr.lib.utils.mdm_utils import viz_motions

from nemo.utils.misc_utils import to_np, to_tensor

from mdm.utils.fixseed import fixseed
from mdm.dataset.HumanML3D_home.human_body_prior.body_model.body_model import BodyModel
from mdm.data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from mdm.data_loaders.humanml.common.skeleton import Skeleton
from mdm.data_loaders.humanml.scripts.motion_process import process_file
from mdm.data_loaders.humanml.scripts import motion_process

from mdm.data_loaders.get_data import get_dataset_loader

DIR_HUMANML3D_HOME = os.path.join(os.environ['BIO_POSE_ROOT'], 'mdm',
                                  'dataset', 'HumanML3D_home')
DIR_HUMANML3D = os.path.join(os.environ['BIO_POSE_ROOT'], 'mdm', 'dataset',
                             'HumanML3D')
from nemo.utils.misc_utils import to_np, to_tensor


class HumamnMlConverter(object):
	def __init__(self):
		self.humanml_mean = np.load(os.path.join(DIR_HUMANML3D, 'Mean.npy'))
		self.humanml_std = np.load(os.path.join(DIR_HUMANML3D, 'Std.npy'))
		##### for `amass_to_pose` function
		self.trans_matrix = np.array([[1.0, 0.0, 0.0],
			                        [0.0, 0.0, 1.0],
			                        [0.0, 1.0, 0.0]])
		self.ex_fps = 20
		self.fps = 0 

		## some joint indices
		# Lower legs
		# self.l_idx1, self.l_idx2 = 5, 8
		# # Right/Left foot
		# self.fid_r, self.fid_l = [8, 11], [7, 10]
		# # Face direction, r_hip, l_hip, sdr_r, sdr_l
		self.face_joint_indx = [2, 1, 17, 16]
		# # l_hip, r_hip
		# self.r_hip, self.l_hip = 2, 1
		self.joints_num = 22
		## skeleton params
		self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
		self.kinematic_chain = t2m_kinematic_chain

		### Pick one sample to be the target skelton ... this is based on absolute positions
		example_id = "000021.npy"
		example_data = np.load(os.path.join(DIR_HUMANML3D_HOME, "joints", example_id))
		example_data = example_data.reshape(len(example_data), -1, 3)
		example_data = torch.from_numpy(example_data)
		self.tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
		self.tgt_offsets = self.tgt_skel.get_offsets_joints(example_data[0])

	def load_body_model(self):
		if hasattr(self, "male_bm"):
			return

		male_bm_path = DIR_HUMANML3D_HOME+'/body_models/smplh/male/model.npz'
		male_dmpl_path = DIR_HUMANML3D_HOME+'/body_models/dmpls/male/model.npz'
		female_bm_path = DIR_HUMANML3D_HOME+'/body_models/smplh/female/model.npz'
		female_dmpl_path = DIR_HUMANML3D_HOME+'/body_models/dmpls/female/model.npz'
		num_betas, num_dmpls = 10, 8
		self.male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path)
		self.female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path)
		return

	def amass_to_joint_positions(self, bdata, remove_traslation=False, comp_device='cpu'):
	    """
		Adapdted from HumanML3D repo, file raw_pose_processing.ipynb. 

	    Take data in AMASS form (a dictionary with pose, shape, and metadata)
	    and convert to joint positions in absolute space (xyz) coordinates
	    Modified from HumanML3D `raw_pose_processing.ipynb`

	    Resample to have fps=20.
	    
	    Args: 
	    bdata: Dict having `poses`, `betas`, `dmpl` that can be used by a pose 
	    	model. Also `traj`, `mocap_framerate`. 
	    remove_traslation (bool): if true, set root translation to 0.

	    Returns: 
			positions: (np.array) shape (T,J,3), where J=22+30=52 (22 body, 30 hands)
				Global xyz positions of the joints.
	    """
	    self.load_body_model()
	    try:
	        fps = bdata['mocap_framerate']
	        frame_number = bdata['trans'].shape[0]
	    except:
	        raise
	    
	    fId = 0 # frame id of the mocap sequence
	    pose_seq = []
	    if bdata['gender'] == 'male':
	        bm = self.male_bm
	    else:
	        bm = self.female_bm
	    down_sample = int(fps / self.ex_fps)
	    
	    with torch.no_grad():
	        for fId in range(0, frame_number, down_sample):
	            root_orient = torch.Tensor(bdata['poses'][fId:fId+1, :3]).to(comp_device) # controls the global root orientation
	            pose_body = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(comp_device) # controls the body
	            pose_hand = torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(comp_device) # controls the finger articulation
	            betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device) # controls the body shape
	            trans = torch.Tensor(bdata['trans'][fId:fId+1]).to(comp_device)   
	            if remove_traslation:  
	            	trans = trans*0
	            body = bm(pose_body=pose_body, betas=betas, root_orient=root_orient, pose_hand=pose_hand)
	            # next line gets the positions of the joints in xyz
	            joint_loc = body.Jtr[0] + trans
	            pose_seq.append(joint_loc.unsqueeze(0))
	    pose_seq = torch.cat(pose_seq, dim=0)
	    
	    pose_seq_np = pose_seq.detach().cpu().numpy()
	    pose_seq_np_n = np.dot(pose_seq_np, self.trans_matrix)
	    
	    return pose_seq_np_n 

	def joint_positions_to_humanml3d(self, positions, num_joints=22, axis_scaling=1,
				 do_z_normalization=True,  motion_mean=None, motion_std=None):
		"""
		Map absolute xyz positions of ONE SAMPLE to the HumanML3d 263-dim representation
		
		Procedure (and warnings about expected data changes): 
			Map xyz positions to joint rotations. 
			Using those rotations, take a template skeleton and compute back the xyz positions. 
			Rotate the body so frame 0 is at the origin facing forwards. 
			Compute the HumanML3D data representation which has the following: 
		
		Args: 
			positions: output of `amass_to_joint_positions` shape (T,J,3) It will just truncate the 
			first 22 joints.  
			reverse_axes: do_z_normalization
			axis_scaling: scaler number - remove
		Output:
			0: data, shape (T-1,263) HumanML3D vectors for the (T-1) frames (one less 
				frame because of the velocity computation)

		Indices
			0: root rotation velocity about y (vertical) axis. 
			{1,2}:  root linear velocity in x and z. 
			{3}: root position in y. 
			Note: idxs 0-4 define the root position as long as you assume the initial root 
				orientation is 0
			4-67: positions of 21 joints relative to the root orientation and position, 63-dims (3*21).
			67-193: 6d angular rotation of the 21 non-root joints, 126=dims (6*21)
			193-259: velocity of 22 joints, 66-dims (3*22) 

		Slicing operations 
			IDXS_ROOT = slice(0,4)			  		# root angular velocity, xz velocity, and y-position, 4-dims. 
			IDXS_ROOT_NONROTATION_TERMS = slice(1,4)		# same as previous, but not including the root angular velocity
			IDXS_REL_POSITIONS = slice(4,67)  		# positions of 21 joints relative to the root, 63-dims (3*21).
			SLC_CONT6D = slice(67,193)				# 6d angular rotation of the 21 non-root joints, 126=dims (6*21)
			SLC_VEL = slice(193, 259) 				# velocity of 22 joints, 66-dims (3*22) 
			SLC_FEET = slice(259,263)	# feet binary indicators 

		"""
		if type(positions)==torch.Tensor: 
			positions=positions.cpu().numpy()

		positions = positions[:,:num_joints] 
		positions = positions/axis_scaling

		data, global_positions, positions, l_velocity = process_file(
			positions=positions, 
			feet_thre=0.002, 
			tgt_offsets=self.tgt_offsets,			#  skeketon proportions using one skeleton exmaple
			n_raw_offsets=self.n_raw_offsets,
			kinematic_chain=self.kinematic_chain,
			)

		if do_z_normalization:
			if (motion_mean is None) or (motion_std is None):
				motion_mean, motion_std = self.humanml_mean, self.humanml_std 
			data = (data - motion_mean) / motion_std

		return data, global_positions, positions, l_velocity

	def humanml3d_to_joint_positions(self, sample, return_root_info=False):
		"""
		Convert humanml3d to absolute joint positions
		
		Args:
		sample: shape (N,263,1,T), humanML vector

		Returns:
		sample: shape (N,22,3,T)
		"""
		sample = self.inv_transform_z_normalization(sample.cpu().permute(0, 2, 3, 1)).float()
		r_rot_quat, r_pos = motion_process.recover_root_rot_pos(sample)  # r_pos is xyz position
		sample = motion_process.recover_from_ric(sample, self.joints_num)
		sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
		
		if return_root_info:
			return sample.numpy(), r_rot_quat, r_pos
		else: 
			return sample.numpy()

	def inv_transform_z_normalization(self, motion,):
		"""
		HumanML data reps in some works (e.g. MDM) z-normalize 
		This is the inverse transformation using the precomputed mean and 
		std of HumanML3d vectors across the whole dataset. 
		Same as `data.dataset.t2m_dataset.inv_transform`

		Args: 
			motion: shape (..., J=263)
		""" 
		return motion * self.humanml_std + self.humanml_mean

	def smplh_to_humanml3d(self, motion):
		"""
		Note: it's lossy bc you lose the hand motion. 

		Args:
			motion: Shape (N,J,3,T). The last element is the trajectory
			in absolute coords
		"""
		raise NotImplementedError

	def humanml3d_to_rot6d(self, motion):
		# SLC_CONT6D = slice(67,193). Need to also get the root orientation. 
		raise NotImplementedError()

	def humanml3d_remove_translation_long_way(self, sample, do_z_normalization=True):
		""" 
		Given a motion in the standard HumanML format and scale that MDM expects (z-normalized), 
		convert back to raw joint positions, and remove the root translation. 
		Then convert back to HumanML3d format. 

		The point is to be confident that the HumanML3d representation does not contain 
		root translation information in any of its 

		Keep in the the same frame rate.
		Args:
			sampleInput motion (N,J=263,1,T)


		"""	
		# recover xyz positions, and root positions
		positions_global, r_rot_quat, r_pos = self.humanml3d_to_joint_positions(sample, return_root_info=True)

		# now remove trajectory
		r_pos = r_pos.permute(0,1,3,2)
		r_pos = r_pos.expand(r_pos.shape[0], self.joints_num, *r_pos.shape[2:]).numpy()
		positions_global_traj_removed = (positions_global - r_pos)

		# convert back to humanml3d. joint_positions_to_humanml3d func expects shape (T,22,3)
		positions_global_traj_removed = torch.Tensor(positions_global_traj_removed).permute(0,3,1,2).numpy()
		data =  np.stack(
			[self.joint_positions_to_humanml3d(p, do_z_normalization=True)[0] for p in positions_global_traj_removed]
		)
		data = torch.Tensor(data).unsqueeze(1).permute((0,3,1,2))

		# finally, copy the last frame and stick it on the end - this is because HumanML3d loses 
		# a dimension from velocity computation, so this keeps the dimensions right
		data = torch.cat((data, data[...,[-1]]), dim=-1)

		return data



def get_sample_amass_data(paths_amass_source, do_z_normalization=True, 
		motion_mean=None, motion_std=None, max_frames=120):
	"""
	Given a list of paths to AMASS original (in SMPL+H format), get the pose sequence in HumanML format. 
	Also create a copy where the root translation is set to 0 while in AMASS format before converting to HumanML. 
	Function is slow (a few second per sample). 

	A lot of details are hidden in the 2 functions of the `SmplhPoseToHumanML3DPose` class. 
	Args: 
		paths_amass_source: list of paths to the `.npz` files in amass format. 
		do_z_normalize: z-normalize each humanml3d vector (this is done during MMD training for example). 
		motion_mean: the mean for z-normalization. If `do_z_normalize=True` it must be set.
		motion_std: the std for z-normalization. If `do_z_normalize=True` it must be set.
		max_frames: all motions are set to this frame length. If shorter, pad 0s. If longer take the first `max_frmames`.
	"""
	pose_converter = HumamnMlConverter()
	motion_ = []
	motion_traj_removed_ = []
	motion_amass_format_ = []
	paths_amass_source_ = []
	motion_positions_from_amass = []

	# default normalization params 
	if do_z_normalization and ((motion_mean is None) or (motion_std is None)): 
		motion_mean = np.load(os.path.join(DIR_HUMANML3D, 'Mean.npy'))
		motion_std = np.load(os.path.join(DIR_HUMANML3D, 'Std.npy'))

	for i in tqdm.trange(len(paths_amass_source)):
		try: 
			motion_amass_format = np.load(paths_amass_source[i])
		except: 
			print("failed to load ", i, paths_amass_source[i])
			continue
		motion_amass_format_.append(motion_amass_format)
		paths_amass_source_.append(paths_amass_source[i])

		motion = pose_converter.amass_to_joint_positions(motion_amass_format, remove_traslation=False)
		motion_positions_from_amass.append(motion.copy())

		motion_traj_removed = pose_converter.amass_to_joint_positions(motion_amass_format, remove_traslation=True)
		motion, global_positions, positions, l_velocity = pose_converter.joint_positions_to_humanml3d(motion, 
			do_z_normalization=do_z_normalization, motion_mean=motion_mean, motion_std=motion_std)
		motion_traj_removed, global_positions_traj_removed, positions_traj_removed, l_velocity_traj_removed = pose_converter.joint_positions_to_humanml3d(motion_traj_removed, 
			do_z_normalization=do_z_normalization, motion_mean=motion_mean, motion_std=motion_std)
		m_length = motion.shape[0]
		motion = motion[:max_frames]
		motion_traj_removed = motion_traj_removed[:max_frames]
		if m_length <max_frames:
			motion = np.concatenate([motion,np.zeros((max_frames - m_length, motion.shape[1]))], axis=0) 
			motion_traj_removed = np.concatenate([motion_traj_removed,np.zeros((max_frames - m_length, motion.shape[1]))], axis=0) 

		motion_.append(motion)
		motion_traj_removed_.append(motion_traj_removed)

	N = len(motion_)    # since some reads will fail
	motion = np.stack(motion_, axis=0)
	motion_traj_removed = np.stack(motion_traj_removed_, axis=0)
	
	# change shape from is (T,266) --> (1,263,1,T)
	motion_humanml = torch.Tensor(motion).permute(0,2,1).unsqueeze(2)
	motion_traj_removed_humanml = torch.Tensor(motion_traj_removed).permute(0,2,1).unsqueeze(2)

	return motion_humanml, motion_traj_removed_humanml, motion_amass_format_, paths_amass_source_, motion_positions_from_amass


if __name__=="__main__":
	DATASET="3dpw"
	log_dir = os.path.join(os.environ['BIO_POSE_ROOT'], "gthmr", "results","conversion_example")
	log_dir = os.path.join(log_dir, DATASET)
	
	if DATASET=="3dpw":
		print("Example conversion using 3dpw data: xyz global positions >> humanml format")
		from VIBE.lib.dataset.threedpw import ThreeDPW
		from torch.utils.data import DataLoader
		from VIBE.lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
		fixseed(0)
		device='cuda'

		### 1: get `smpl_joints`: the xyz positions of 3dpw data using SMPL ###
		batch_size, seqlen = 20, 121
		train_3d_db = ThreeDPW(set='train', seqlen=seqlen, debug=False)
		tmp = train_3d_db[0]
		train_3d_loader = DataLoader(dataset=train_3d_db,batch_size=batch_size, shuffle=True,num_workers=0,)
		target = next(iter(train_3d_loader))
		smpl = SMPL(SMPL_MODEL_DIR, batch_size=batch_size, create_transl=False).to(device)
		theta = target['theta'][:, :, 3:75].reshape(-1, 72).to(device)
		zero_betas = torch.zeros((theta.shape[0], 10)).cuda()
		with torch.no_grad():
			gt_output = smpl(betas=zero_betas,body_pose=theta[:, 3:],global_orient=theta[:, :3],pose2rot=True)
			smpl_joints = gt_output.smpl_joints
			smpl_joints = smpl_joints.reshape(batch_size, seqlen, 45,3)[:, :, :24]  # (N, T, 24, 3)

		### 2: visualize original 3dpw data in directory gthmr/results/conversion_example/3dpw_format ##
		log_dir_3dpw = os.path.join(log_dir, "3dpw_format")
		os.makedirs(log_dir_3dpw, exist_ok=True)
		smpl_joints_postprocessed = smpl_joints.permute(0,2,3,1).cpu().numpy()
		viz_motions(batch_size, 1, log_dir_3dpw, smpl_joints_postprocessed, dataset='humanact12')
			
		### 3: get `data_humanml`: convert `smpl_joints` xyz positions to HumamnML vector 
		# 		 using `HumamnMlConverter.joint_positions_to_humanml3d` function ###
		converter=HumamnMlConverter()
		smpl_joints = to_np(smpl_joints) # should be (N,T,24,3)
		data_humanml = np.stack([
			converter.joint_positions_to_humanml3d(g, do_z_normalization=True, axis_scaling=-1.5,)[0] 
			for g in smpl_joints
			]) # shape (N,T,263)
		data_humanml=torch.Tensor(data_humanml).permute(0,2,1).unsqueeze(2) # back to shape (N,263,1,T)

		### 4: convert `data_humanml back to to xyz positions and visualise it ###
		data_humanml_postprocessed = converter.humanml3d_to_joint_positions(data_humanml) # (N,22,3,T) 
		batch_size=len(data_humanml_postprocessed)
		log_dir_humanml = os.path.join(log_dir, "humanml_format")
		os.makedirs(log_dir_humanml, exist_ok=True)
		viz_motions(batch_size, 1, log_dir_humanml, data_humanml_postprocessed, dataset='humanml3d')

		import ipdb; ipdb.set_trace()

