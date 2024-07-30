def get_pos_from_rep(model, sample):
    T = sample.shape[-1]
    N = sample.shape[0]
    j_dic = model.forward_kinematics(sample, None)
    smpl_joints = j_dic['kp_45_joints'][:, :22].reshape(N, T, 22, 3)  +  j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)
    return smpl_joints.cpu().numpy()

