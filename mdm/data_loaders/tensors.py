import ipdb
import torch


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(
        lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(
            1)  # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    if 'frame' in notnone_batches[0]:
        framebatch = [b['frame'] for b in notnone_batches]
        cond['y'].update({'frame': torch.as_tensor(framebatch).unsqueeze(1)})

    if "freeze" in notnone_batches[0]:
        freezebatch = [b['freeze'] for b in notnone_batches]
        cond['y'].update({'freeze': torch.stack(freezebatch)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text'] for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    # video features are for cond_mode=='video'
    if 'features' in notnone_batches[0]:
        features = [torch.tensor(b['features']) for b in notnone_batches]
        cond['y'].update({'features': torch.stack(features)})  # (T,featdim)


    if 'cv_pose_6d' in notnone_batches[0]:
        cv_pose_6d_batch = [b['cv_pose_6d'] for b in notnone_batches]
        cv_pose_6d_batch = collate_tensors(cv_pose_6d_batch)
        cond['y'].update({'cv_pose_6d': cv_pose_6d_batch})

    if 'joints3D' in notnone_batches[0]:
        joints3D = [torch.tensor(b['joints3D']) for b in notnone_batches]
        cond['y'].update({'joints3D': torch.stack(joints3D)})  # (T, 14, 3)

    if 'gt_spin_joints3d' in notnone_batches[0]:
        joints3D = [torch.tensor(b['gt_spin_joints3d']) for b in notnone_batches]
        cond['y'].update({'gt_spin_joints3d': torch.stack(joints3D)})  # (T, 14, 3)
    
    # video name for vibe datasets
    if 'vid_name' in notnone_batches[0]:
        import numpy as np  # this data is strings.
        # import ipdb; ipdb.set_trace()
        vid_names = [np.array(b['vid_name']) for b in notnone_batches]
        cond['y'].update({'vid_name': np.stack(vid_names)})  # (T,featdim)

    if 'keyframe' in notnone_batches[0]:
        import numpy as np  # this data is strings.
        # import ipdb; ipdb.set_trace()
        vid_names = [np.array(b['keyframe']) for b in notnone_batches]
        cond['y'].update({'keyframe': np.stack(vid_names)})  # (T,featdim)

    if 'keyframe_mask' in notnone_batches[0]:
        import numpy as np  # this data is strings.
        # import ipdb; ipdb.set_trace()
        vid_names = [np.array(b['keyframe_mask']) for b in notnone_batches]
        cond['y'].update({'keyframe_mask': np.stack(vid_names)})  # (T,featdim)

    if 'img_name' in notnone_batches[0]:
        import numpy as np  # this data is strings.
        # import ipdb; ipdb.set_trace()
        img_names = [b['img_name'] for b in notnone_batches]
        cond['y'].update({'img_name': np.stack(img_names)})  # (N, T, 1)

    if 'kp_2d' in notnone_batches[0]:
        kp_2d = [torch.tensor(b['kp_2d']) for b in notnone_batches]
        cond['y'].update({'kp_2d': torch.stack(kp_2d)}) 

    if 'theta' in notnone_batches[0]:
        theta = [torch.tensor(b['theta']) for b in notnone_batches]
        cond['y'].update({'theta': torch.stack(theta)})  

    if 'trans' in notnone_batches[0]:
        theta = [torch.tensor(b['trans']) for b in notnone_batches]
        cond['y'].update({'trans': torch.stack(theta)})  

    return motion, cond


# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [
        {
            'inp': torch.tensor(
                b[4].T).float().unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2],  #b[0]['caption']
            'tokens': b[6],
            'lengths': b[5],
        } for b in batch
    ]
    return collate(adapted_batch)
