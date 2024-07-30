from torch.utils.data import DataLoader
from mdm.data_loaders.tensors import collate as all_collate
from mdm.data_loaders.tensors import t2m_collate  #, hmr_collate
import ipdb
import copy


def get_dataset_class(name):
    if "amass" in name and "pair" in name:
        from VIBE.lib.dataset import pairs_dataset
        return pairs_dataset.PairsDataset
    elif "amass" in name and "keyframe" in name:
        from VIBE.lib.dataset import keyframe_dataset
        return keyframe_dataset.KeyframeDataset
    elif "amass" in name or name in ['h36m', '3dpw', 'nemomocap', 'nemomocap2']:
        from VIBE.lib.dataset import vibe_dataset
        return vibe_dataset.VibeDataset
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train', task='motion'):
    assert task in ['motion', 'hmr']
    if task == 'motion':
        if hml_mode == 'gt':
            from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
            return t2m_eval_collate
        if name in ["humanml", "kit"]:
            return t2m_collate
        else:
            return all_collate
    else:
        return hmr_collate


def get_dataset(name,
                num_frames,
                split='train',
                hml_mode='train',
                no_motion_augmentation=False,
                rotation_augmentation=False,
                augment_camview=False,
                sideline_view=False,
                data_rep='rot6d'):
    if name != 'amass':
        rotation_augmentation = False
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split,
                       num_frames=num_frames,
                       mode=hml_mode,
                       no_motion_augmentation=no_motion_augmentation)
    elif 'amass' in name:
        if name == "amass_hml_pair" or name == "amass_hml_keyframe":
            name = "amass_hml"
        # these use the same Dataset class
        if ('amass' in name) and name not in ("amass", "amass_hml"):
            # Case where we take subsets of amass. Expect subdataset structure to be like "amass:KIT,CMU,HumanEva"
            restrict_subsets = name[6:].split(",")
            name = 'amass'
        else:
            restrict_subsets = None
        
        dataset = DATA(split=split,
                       num_frames=num_frames,
                       dataset=name,
                       rotation_augmentation=rotation_augmentation,
                       augment_camview=augment_camview,
                       restrict_subsets=restrict_subsets,
                       data_rep=data_rep)
    elif name in ('h36m', '3dpw', 'nemomocap', 'nemomocap2'):
        dataset = DATA(split=split,
                       num_frames=num_frames,
                       rotation_augmentation=rotation_augmentation,
                       augment_camview=augment_camview,
                       restrict_subsets=None,
                       dataset=name,
                       sideline_view=sideline_view,
                       data_rep=data_rep)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name,
                       batch_size,
                       num_frames,
                       split='train',
                       hml_mode='train',
                       no_motion_augmentation=False,
                       num_workers=0,
                       task='motion',
                       shuffle=True,
                       sideline_view=False,
                       rotation_augmentation=False,
                       drop_last=True,
                       augment_camview=False,
                       data_rep='rot6d',
                       **kwargs):
    dataset = get_dataset(name,
                          num_frames,
                          split,
                          hml_mode,
                          no_motion_augmentation=no_motion_augmentation,
                          rotation_augmentation=rotation_augmentation,
                          augment_camview=augment_camview,
                          sideline_view=sideline_view,
                          data_rep=data_rep)
    collate = get_collate_fn(name, hml_mode, task)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        collate_fn=collate)

    return loader

def get_dataset_loader_dict_new2(data_cfg, eval_only=False):
    """
    A even newer version of `get_dataset_loader_dict_new` with multiple training datasets. 

    """
    # Parse kwargs
    kwargs_dict = {}
    for k in data_cfg:
        if k not in ['train_sets', 'eval_sets']:
            kwargs_dict[k] = data_cfg[k]
    # Return values
    train_data_loaders_dic = {'hmr': {}, 'motion': {}}
    eval_data_loaders_dic = {'hmr': {}}

    for typ in ['hmr', 'motion']:
        if data_cfg['train_sets'][typ] is None:
            data_cfg['train_sets'][typ] = []

    total_batch_size = 0
    if not eval_only:
        for typ in ['hmr', 'motion']:
            for train_set in data_cfg['train_sets'][typ]:
                curr_kwargs = copy.deepcopy(kwargs_dict)
                train_set_name = list(train_set.keys())[0]
                train_set_values = list(train_set.values())[0]
                # curr_kwargs['split'] = train_set_values['split']
                for _k, _v in train_set_values.items():
                    curr_kwargs[_k] = _v                
                if typ == 'motion' and 'augment_camview' in train_set_values.keys(
                ):
                    curr_kwargs['augment_camview'] = train_set_values[
                        'augment_camview']
                total_batch_size += curr_kwargs['batch_size']
                loader = get_dataset_loader(train_set_name, **curr_kwargs)
                train_data_loaders_dic[typ][train_set_name] = loader

    # Load eval loader
    if data_cfg['eval_sets']['hmr'] is not None:
        for eval_set in data_cfg['eval_sets']['hmr']:
            curr_kwargs = copy.deepcopy(kwargs_dict)
            eval_set_name = list(eval_set.keys())[0]
            eval_set_values = list(eval_set.values())[0]
            split = eval_set_values['split']
            curr_kwargs['split'] = split
            if 'eval_batch_size' in curr_kwargs:
                curr_kwargs['batch_size'] = curr_kwargs['eval_batch_size']
            loader = get_dataset_loader(eval_set_name, **curr_kwargs)
            eval_data_loaders_dic['hmr'][f"{eval_set_name}_{split}"] = loader

    return train_data_loaders_dic, eval_data_loaders_dic, total_batch_size

def get_dataset_loader_dict_new(data_cfg, eval_only=False):
    raise ValueError()
    # """
    # A new version of `get_dataset_loader_dict` with a list of evaluation datasets.

    # """
    # # Parse kwargs
    # kwargs_dict = {}
    # for k in data_cfg:
    #     if k not in ['train_sets', 'eval_sets']:
    #         kwargs_dict[k] = data_cfg[k]

    # # Return values
    # train_data_loaders_dic = {'hmr': None, 'motion': None}
    # eval_data_loaders_dic = {'hmr': {}}

    # for typ in ['hmr', 'motion']:
    #     if data_cfg['train_sets'][typ] is None:
    #         data_cfg['train_sets'][typ] = []
    # # Load each training set loader
    # # THIS IS WRONG... change this to use ConcatDataset later
    # assert len(data_cfg['train_sets']['hmr']) <= 1
    # assert len(data_cfg['train_sets']['motion']) <= 1

    # if not eval_only:
    #     for typ in ['hmr', 'motion']:
    #         for train_set in data_cfg['train_sets'][typ]:
    #             curr_kwargs = copy.deepcopy(kwargs_dict)
    #             train_set_name = list(train_set.keys())[0]
    #             train_set_values = list(train_set.values())[0]
    #             curr_kwargs['split'] = train_set_values['split']
    #             if typ == 'motion' and 'augment_camview' in train_set_values.keys(
    #             ):
    #                 curr_kwargs['augment_camview'] = train_set_values[
    #                     'augment_camview']

    #             loader = get_dataset_loader(train_set_name, **curr_kwargs)
    #             train_data_loaders_dic[typ] = loader

    # # Load eval loader
    # if data_cfg['eval_sets']['hmr'] is not None:
    #     for eval_set in data_cfg['eval_sets']['hmr']:
    #         curr_kwargs = copy.deepcopy(kwargs_dict)
    #         eval_set_name = list(eval_set.keys())[0]
    #         eval_set_values = list(eval_set.values())[0]
    #         split = eval_set_values['split']
    #         curr_kwargs['split'] = split
    #         if 'eval_batch_size' in curr_kwargs:
    #             curr_kwargs['batch_size'] = curr_kwargs['eval_batch_size']
    #         loader = get_dataset_loader(eval_set_name, **curr_kwargs)
    #         eval_data_loaders_dic['hmr'][f"{eval_set_name}_{split}"] = loader

    # return train_data_loaders_dic, eval_data_loaders_dic


def get_dataset_loader_dict(data_cfg, eval_only=False):
    raise ValueError()
    # Use `get_dataset_loader_dict_eval_list` instead

    # # Parse kwargs
    # kwargs_dict = {}
    # for k in data_cfg:
    #     if k not in ['train_sets', 'eval_sets']:
    #         kwargs_dict[k] = data_cfg[k]

    # # Return values
    # train_data_loaders_dic = {'hmr': None, 'motion': None}
    # eval_data_loaders_dic = {'hmr': None}

    # for typ in ['hmr', 'motion']:
    #     if data_cfg['train_sets'][typ] is None:
    #         data_cfg['train_sets'][typ] = []
    # # Load each training set loader
    # # THIS IS WRONG... change this to use ConcatDataset later
    # assert len(data_cfg['train_sets']['hmr']) <= 1
    # assert len(data_cfg['train_sets']['motion']) <= 1

    # if not eval_only:
    #     for typ in ['hmr', 'motion']:
    #         for train_set in data_cfg['train_sets'][typ]:
    #             curr_kwargs = copy.deepcopy(kwargs_dict)
    #             train_set_name = list(train_set.keys())[0]
    #             train_set_values = list(train_set.values())[0]
    #             curr_kwargs['split'] = train_set_values['split']
    #             if typ=='motion' and 'augment_camview' in train_set_values.keys():
    #               curr_kwargs['augment_camview'] = train_set_values['augment_camview']

    #             loader = get_dataset_loader(train_set_name, **curr_kwargs)
    #             train_data_loaders_dic[typ] = loader

    # # Load eval loader
    # if data_cfg['eval_sets']['hmr'] is not None:
    #   eval_set = data_cfg['eval_sets']['hmr'][0]
    #   curr_kwargs = copy.deepcopy(kwargs_dict)
    #   eval_set_name = list(eval_set.keys())[0]
    #   eval_set_values = list(eval_set.values())[0]
    #   curr_kwargs['split'] = eval_set_values['split']
    #   if 'eval_batch_size' in curr_kwargs:
    #       curr_kwargs['batch_size'] = curr_kwargs['eval_batch_size']
    #   loader = get_dataset_loader(eval_set_name, **curr_kwargs)
    #   eval_data_loaders_dic['hmr'] = loader

    # return train_data_loaders_dic, eval_data_loaders_dic
