import torch


import numpy as np
import mmcv
from mmcv.image import imread, imwrite
from mmdet.core import get_classes

import mmcv
from mmcv.runner import load_checkpoint, parallel_test
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet.models import build_detector, detectors

import dataikuapi as dka
import pandas as pd
import cv2
import os,time
import os.path as osp

from mmdet.apis import inference_detector
from mmdet.datasets import build_dataloader

from dss.api import read_dataframe, write_dataframe
from environment import ROOT_DIR, TMP_DIR, DSS_DIR, API_KEY_VIT, PROJECT_KEY_VIT, DSS_HOST, VERTICA_HOST
from generator_detection import CustomDataset

def load_data(dataset_name = 'img_MIF',nrows=1e3):
    # cached the dataset as csv
    dataset_path = os.path.join(DSS_DIR,'{}.csv'.format(dataset_name))
    if not os.path.isfile(dataset_path):

        img_MIF_df = read_dataframe(API_KEY_VIT,VERTICA_HOST,PROJECT_KEY_VIT,dataset_name, columns=['path','img1','img2'])
        img_MIF_df.to_csv(dataset_path)
    else:
        print('Read cached csv')
        img_MIF_df = pd.read_csv(dataset_path, nrows=nrows)


    print(img_MIF_df.head())
    print('%s rows have been retrieved'%img_MIF_df.shape[0])

    img_MIF_df = img_MIF_df.assign(
    img1_path=(ROOT_DIR + img_MIF_df['path'] + "/" + img_MIF_df['img1']),
    img2_path=(ROOT_DIR + img_MIF_df['path'] + "/" + img_MIF_df['img2']),
                    )
    img_df = pd.melt(img_MIF_df,
        id_vars='path',
        value_vars=['img1','img2'], # list of days of the week
        var_name='img_num',
        value_name='img_name').sort_values('path')

    img_df = img_df[(img_df.img_name != "") & (img_df.path != "")]
    img_df.dropna(subset=['path','img_name'],inplace=True,how='any')

    img_df.reset_index(inplace=True)

    return img_df


def load_config(modele):

    ## PREPARE FROM MMCV-DET
    cfg = mmcv.Config.fromfile('/usr/src/app/configs/%s.py'%modele['conf'])
    cfg.model.pretrained = None
    return cfg


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results



def det_bboxes(bboxes,
              labels,
              class_names=None,
              class_to_keep=None,
              score_thr=0,
              ):
    """Save bboxes and class labels (with scores) on an image.
    Args:
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        class_to_keep (list[str]): Classes to keep (cars, trucks...)

        score_thr (float): Minimum score of bboxes to be shown.

    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    to_save = []


    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)

        if label_text in class_to_keep:
            to_save.append({'x1':bbox_int[0],'y1':bbox_int[1],
                            'x2':bbox_int[2],'y2':bbox_int[3],
                            'class':label_text,'score':bbox[-1]})

    return to_save

def save_result(result,
                class_to_keep=[],
                dataset='coco',
                score_thr=0.3
                ):

    """Return list dict [{x1,x2,y1,y2,classe,score},...]
    Args:
        results:
        class_to_keep (list[str]): Classes to keep (cars, trucks...)
        score_thr (float): Minimum score of bboxes to be shown.
    """
    class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)

    return  det_bboxes(
        bboxes,
        labels,
        class_names=class_names,
        class_to_keep=class_to_keep,
        score_thr=score_thr)

modele = dict(conf="retinanet_r50_fpn_1x",
              checkpoint="retinanet_r50_fpn_1x_20181125-3d3c2142")
modele = dict(conf="retinanet_x101_64x4d_fpn_1x",
          checkpoint="retinanet_x101_64x4d_fpn_1x_20181218-2f6f778b")



def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def test_CustomDataset():
    img_df = load_data()
    cfg = load_config(modele)
    print(img_df.shape[0])
    car_dataset = CustomDataset(img_df, ROOT_DIR,**cfg.data.val)

    print(len(car_dataset))

    first_element = car_dataset[0]
    print(first_element)

def main():
    # Build dataset

    img_df = load_data(nrows=500)
    cfg = load_config(modele)
    car_dataset = CustomDataset(img_df, ROOT_DIR,**cfg.data.val)
    print('length of dataset :')
    print(len(car_dataset))
    class_to_keep = ['person','bicycle', 'car',
                'motorcycle','bus',
                'truck','traffic_light','stop_sign',
                'parking_meter','bench']

    # build modele

    gpus = 4
    workers_per_gpu = 2

    checkpoint = '/model/%s.pth'%modele['checkpoint']

    if gpus == 1:
        show = False
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        _ = load_checkpoint(model, checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            car_dataset,
            imgs_per_gpu=1,
            workers_per_gpu=workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)

        outputs = single_test(model, data_loader, show)

    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            checkpoint,
            car_dataset,
            _data_func,
            range(gpus),
            workers_per_gpu=workers_per_gpu)


    for row, results  in zip(img_df.iterrows(),outputs):
        to_save = save_result(results,
                    class_to_keep=class_to_keep,
                    dataset='coco',
                    score_thr=0.3)



if __name__ == '__main__':

    main()
    """

    #device = 0
    num_gpu = 4

    modele = dict(conf="retinanet_r50_fpn_1x",
                  checkpoint="retinanet_r50_fpn_1x_20181125-3d3c2142")
    modele = dict(conf="retinanet_x101_64x4d_fpn_1x",
              checkpoint="retinanet_x101_64x4d_fpn_1x_20181218-2f6f778b")
    ## PREPARE FROM MMCV-DET
    cfg = mmcv.Config.fromfile('/usr/src/app/configs/%s.py'%modele['conf'])
    cfg.model.pretrained = None


    class_to_keep = ['person','bicycle', 'car',
                    'motorcycle','bus',
                    'truck','traffic_light','stop_sign',
                    'parking_meter','bench']


    # Build dataset


    ## one gpus
    # build modele
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, '/model/%s.pth'%modele['checkpoint'])
    model = MMDataParallel(model, device_ids=[0])
    ## END OF PREPARE FROM DSS
    """
    """
    def send_to_dss(args):

        device = args[0]
        print(device)
        img_df = args[1]
        ## INFERENCE
        # test a single image
        t1 = time.time()
        # construct the model and load checkpoint
        model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
        #model.to(device)# = MMDataParallel(model, device_ids=range(num_gpu))
        _ = load_checkpoint(model, '/model/%s.pth'%modele['checkpoint'])


        ## END OF PREPARE FROM MMCV-DET
        col_seg = ['img_name','path','x1','y1','x2','y2',
                                        'class','score']
        img_seg = pd.DataFrame(columns=col_seg)
        log_seg = pd.DataFrame(columns=['img_name','path','traceback'])

        for i,(_,row) in enumerate(img_df.iterrows()):

            traceback = ''
            img_path = os.path.join(ROOT_DIR,row['path'],row['img_name'])

            try:
                img = mmcv.imread(img_path)
                result = inference_detector(model, img, cfg, device='cuda:%s'%device)
                to_save = save_result(result,
                            class_to_keep=class_to_keep,
                            dataset='coco',
                            score_thr=0.3)
                to_save_df = pd.DataFrame(to_save)
                to_save_df['img_name'] = row['img_name']
                to_save_df['path'] = row['path']

                img_seg = pd.concat([img_seg, to_save_df],
                        ignore_index=True,sort=False)


            except Exception as traceback:
                # for # DEBUG:
                print(img_path)
                print(traceback)

            log_seg.loc[i,'img_name']  = row['img_name']
            log_seg.loc[i,'path']  = row['path']
            log_seg.loc[i,'traceback']  = traceback
            # ax[i]


            if i%1000 == 0:

                img_seg['score'] = img_seg['score'].round(decimals=2)

                print('%s images processed on the device %s'%(i,device))
                write_dataframe(VERTICA_HOST,PROJECT_KEY_VIT,'log_%s'%modele['conf'],log_seg)
                write_dataframe(VERTICA_HOST,PROJECT_KEY_VIT,'box_%s'%modele['conf'],img_seg[col_seg])

                # flush the results dataframe
                img_seg.drop(img_seg.index, inplace=True)
                log_seg.drop(log_seg.index, inplace=True)
                assert img_seg.shape[0]<1,'pb in flush'



        t2 = time.time()
        print('Executed in %0.d s on the device %s'%((t2-t1),device))
        ## END INFERENCE

        ## END SAVE TO DSS
        return device


    send_to_dss((0,img_df))

    """
