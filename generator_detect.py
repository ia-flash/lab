# from mmdetection/apis/inference.py
def _prepare_data(img, img_transform, cfg, device):

    return dict(img=[img], img_meta=[img_meta])

class FlashDataset(Dataset):
    """Flashed vehicles dataset."""

    def __init__(self, csv_file, root_dir,cfg,device=1):
        """
        Args:
            img_df (DataFrame)
            root_dir : Absolute path to the img
        """
        self.img_df = img_df.copy(deep=True)
        self.img_df.reset_index(inplace=True)
        self.root_dir = root_dir
        self.cfg = cfg
        self.img_transform = ImageTransform(
            size_divisor=cfg.data.test.size_divisor,
            **cfg.img_norm_cfg)

        for col in ['img_name','path']:
            assert col in self.img_df.columns

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.img_df.loc[idx, 'path'],
                                self.img_df.loc[idx, 'img_name'])

        img = mmcv.imread(img_name)

        ori_shape = img.shape
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, scale=cfg.data.test.img_scale)
        img = to_tensor(img).to(device)
        img_meta = [
            dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=False)
        ]

        data = dict(img=imgs, img_meta=img_metas)

        return data

if __name__ == '__main__':

    dataset1 = FlashDataset(img_df,root_dir)
    for i in range(10):
        print(dataset1[i].shape)
