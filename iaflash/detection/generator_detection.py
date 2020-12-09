import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

class CustomDataset(Dataset):
    def __init__(self,img_df, root_dir,pipeline,
                type=None,ann_file=None,img_prefix=None,
                img_scale=(1333, 800),
                size_divisor=32,
                flip_ratio=0,
                with_mask=False,
                with_crowd=False,
                with_label=True,
                img_norm_cfg=None,
                resize_keep_ratio=True):

        self.img_df = img_df
        self.root_dir = root_dir
        self.size_divisor = size_divisor
        self.img_norm_cfg = img_norm_cfg
        self.resize_keep_ratio = resize_keep_ratio
        self.flip_ratio = flip_ratio

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]


        assert mmcv.is_list_of(self.img_scales, tuple)

        self.pipeline = Compose(pipeline)

    def __len__(self):
        return self.img_df.shape[0]

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        row =  self.img_df.iloc[idx]
        img_path = osp.join(self.root_dir,row['path'],row['img_name'])

        img = mmcv.imread(img_path)
        img_info = {'height':img.shape[0] , 'width': img.shape[1], 'filename':img_path}
        results = dict(img_info=img_info,
                    img_prefix=osp.join(self.root_dir,row['path']))
        #print(img_info)
        return self.pipeline(results)


    def __getitem__(self, idx):

        return self.prepare_test_img(idx)
