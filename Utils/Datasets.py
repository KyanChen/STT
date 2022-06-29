import os.path

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from skimage import io
from Utils.Augmentations import Augmentations, Resize


class Datasets(Dataset):
    def __init__(self, data_file, transform=None, phase='train', *args, **kwargs):
        self.transform = transform
        self.data_info = pd.read_csv(data_file, index_col=0)
        self.phase = phase

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        data = self.pull_item_seg(index)
        return data

    def pull_item_seg(self, index):
        """
        :param index: image index
        """
        data = self.data_info.iloc[index]
        img_name = data['img']
        label_name = data['label']

        ori_img = io.imread(img_name, as_gray=False)
        ori_label = io.imread(label_name, as_gray=True)
        assert (ori_img is not None and ori_label is not None), f'{img_name} or {label_name} is not valid'

        if self.transform is not None:
            img, label = self.transform((ori_img, ori_label))

        one_hot_label = np.zeros([2] + list(label.shape), dtype=np.float)
        one_hot_label[0] = label == 0
        one_hot_label[1] = label > 0
        return_dict = {
            'img': torch.from_numpy(img).permute(2, 0, 1),
            'label': torch.from_numpy(one_hot_label),
            'img_name': os.path.basename(img_name)
        }
        return return_dict


def get_data_loader(config, test_mode=False):
    if not test_mode:
        train_params = {
            'batch_size': config['BATCH_SIZE'],
            'shuffle': config['IS_SHUFFLE'],
            'drop_last': False,
            'collate_fn': collate_fn,
            'num_workers': config['NUM_WORKERS'],
            'pin_memory': False
        }
        #  data_file, config, transform=None
        train_set = Datasets(
            config['DATASET'],
            Augmentations(
                config['IMG_SIZE'], config['PRIOR_MEAN'], config['PRIOR_STD'], 'train', config['PHASE'], config
            ),
            config['PHASE'],
            config
        )
        patterns = ['train']
    else:
        patterns = []

    if config['IS_VAL']:
        val_params = {
            'batch_size': config['VAL_BATCH_SIZE'],
            'shuffle': False,
            'drop_last': False,
            'collate_fn': collate_fn,
            'num_workers': config['NUM_WORKERS'],
            'pin_memory': False
        }
        val_set = Datasets(
            config['VAL_DATASET'],
            Augmentations(
                config['IMG_SIZE'], config['PRIOR_MEAN'], config['PRIOR_STD'], 'val', config['PHASE'], config
            ),
            config['PHASE'],
            config
        )
        patterns += ['val']

    if config['IS_TEST']:
        test_params = {
            'batch_size': config['VAL_BATCH_SIZE'],
            'shuffle': False,
            'drop_last': False,
            'collate_fn': collate_fn,
            'num_workers': config['NUM_WORKERS'],
            'pin_memory': False
        }
        test_set = Datasets(
            config['TEST_DATASET'],
            Augmentations(
                config['IMG_SIZE'], config['PRIOR_MEAN'], config['PRIOR_STD'], 'test', config['PHASE'], config
            ),
            config['PHASE'],
            config
        )
        patterns += ['test']

    data_loaders = {}
    for x in patterns:
        data_loaders[x] = DataLoader(eval(x+'_set'), **eval(x+'_params'))
    return data_loaders


def collate_fn(batch):
    def to_tensor(item):
        if torch.is_tensor(item):
            return item
        elif isinstance(item, type(np.array(0))):
            return torch.from_numpy(item).float()
        elif isinstance(item, type('0')):
            return item
        elif isinstance(item, list):
            return item
        elif isinstance(item, dict):
            return item

    return_data = {}
    for key in batch[0].keys():
        return_data[key] = []

    for sample in batch:
        for key, value in sample.items():
            return_data[key].append(to_tensor(value))

    keys = set(batch[0].keys()) - {'img_name'}
    for key in keys:
        return_data[key] = torch.stack(return_data[key], dim=0)

    return return_data

