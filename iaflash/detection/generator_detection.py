import mmcv
import os
from torch.utils.data import Dataset

from mmcv.parallel import DataContainer as DC

from mmdet.datasets.transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)

from mmdet.datasets.utils import to_tensor

class CustomDataset(Dataset):
    def __init__(self,img_df, root_dir,
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

        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)


    def __len__(self):
        return self.img_df.shape[0]

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        row =  self.img_df.iloc[idx]
        img_path = os.path.join(self.root_dir,row['path'],row['img_name'])

        img = mmcv.imread(img_path)
        img_info = {'height':img.shape[0] , 'width': img.shape[1]}
        #print(img_info)

        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                img_name=row['img_name'],
                path=row['path'])


            return _img, _img_meta

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(
                img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(
                    img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas)

        return data


    def __getitem__(self, idx):

        return self.prepare_test_img(idx)
