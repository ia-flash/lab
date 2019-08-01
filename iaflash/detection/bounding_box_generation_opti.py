import torch


import numpy as np
import mmcv
from mmcv.image import imread, imwrite

import mmcv
from mmcv.runner import load_checkpoint#, parallel_test
from mmcv.parallel import scatter, collate, MMDataParallel, MMDistributedDataParallel

from mmdet.models import build_detector, detectors

import pandas as pd
import cv2
import os,time
import os.path as osp

from mmdet.apis import inference_detector, init_detector, init_dist
from mmdet.datasets import build_dataloader

from iaflash.dss.api import read_dataframe, write_dataframe
from iaflash.environment import ROOT_DIR, TMP_DIR, DSS_DIR, API_KEY_VIT, PROJECT_KEY_VIT, DSS_HOST, VERTICA_HOST
from iaflash.detection.generator_detection import CustomDataset
from iaflash.detection.utils import load_data, chunker, save_result
from mmdet.core import get_classes


class_to_keep = ['person','bicycle', 'car',
                'motorcycle','bus',
                'truck','traffic_light','stop_sign',
                'parking_meter','bench']

col_seg = ['img_name','path','x1','y1','x2','y2',
                                    'class','score']

modele = dict(conf="retinanet_r50_fpn_1x",
              checkpoint="retinanet_r50_fpn_1x_20181125-7b0c2548")




def load_config(modele):

    ## PREPARE FROM MMCV-DET
    cfg = mmcv.Config.fromfile('/workspace/mmdetection/configs/%s.py'%modele['conf'])
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    return cfg


def single_test(model, data_loader, show=False):
    model.eval()
    results, logs= [], []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        traceback=''
        try :
            with torch.no_grad():
                result = model(return_loss=False, rescale=not show, **data)


            if show:
                model.module.show_result(data, result, dataset.img_norm_cfg,
                                         dataset=dataset.CLASSES)

        except Exception as traceback:
            print(traceback)
            result = []

        logs.append(traceback)
        results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()


    return results, logs


import multiprocessing
def worker_func(model_cls, model_kwargs, dataset, data_func,
                gpu_id, idx_queue, result_queue):

    config_file = os.path.join('/workspace/mmdetection/configs', f"{modele['conf']}.py")
    checkpoint_file = os.path.join('/model', f"{modele['checkpoint']}.pth")

    model = init_detector(config_file, checkpoint_file)

    #model = model_cls(**model_kwargs)
    #load_checkpoint(model, checkpoint_file, map_location='cpu')
    torch.cuda.set_device(gpu_id)
    model.cuda()
    model.eval()
    with torch.no_grad():
        while True:
            idx = idx_queue.get()
            data = dataset[idx]
            log = ''
            try:
                result = model(**data_func(data, gpu_id))
            except Exception as log:
                result = []

            result_queue.put((idx, result, log))


def parallel_test(model_cls,
                  model_kwargs,
                  dataset,
                  data_func,
                  gpus,
                  workers_per_gpu=1):
    """Parallel testing on multiple GPUs.
    Args:
        model_cls (type): Model class type.
        model_kwargs (dict): Arguments to init the model.
        checkpoint (str): Checkpoint filepath.
        dataset (:obj:`Dataset`): The dataset to be tested.
        data_func (callable): The function that generates model inputs.
        gpus (list[int]): GPU ids to be used.
        workers_per_gpu (int): Number of processes on each GPU. It is possible
            to run multiple workers on each GPU.
    Returns:
        list: Test results.
    """
    ctx = multiprocessing.get_context('spawn')
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = len(gpus) * workers_per_gpu
    workers = [
        ctx.Process(
            target=worker_func,
            args=(model_cls, model_kwargs, dataset, data_func,
                  gpus[i % len(gpus)], idx_queue, result_queue))
        for i in range(num_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    for i in range(len(dataset)):
        idx_queue.put(i)

    results = [None for _ in range(len(dataset))]
    logs = [None for _ in range(len(dataset))]

    prog_bar = mmcv.ProgressBar(task_num=len(dataset))
    for _ in range(len(dataset)):
        idx, res, log = result_queue.get()
        results[idx] = res
        logs[idx] = log
        prog_bar.update()
    print('\n')
    for worker in workers:
        worker.terminate()

    return results, logs




def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def test_CustomDataset():
    img_df = load_data('vit_files_img_xml_trunc',nrows=1e2)
    cfg = load_config(modele)
    print(img_df.shape[0])
    car_dataset = CustomDataset(img_df, ROOT_DIR,**cfg.data.val)

    print(len(car_dataset))

    first_element = car_dataset[0]
    print(first_element)

def run_detection(car_dataset, gpus ,workers_per_gpu, modele):

    config_file = os.path.join('/workspace/mmdetection/configs', f"{modele['conf']}.py")
    checkpoint_file = os.path.join('/model/retina', f"{modele['checkpoint']}.pth")
    cfg = load_config(modele)

    model = init_detector(config_file, checkpoint_file)

    data_loader = build_dataloader(
        car_dataset,
        imgs_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        num_gpus=1,
        dist= (gpus >= 1),
        shuffle=False)

    if gpus == 1:
        show = False
        #_ = load_checkpoint(model, checkpoint)
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_test(model, data_loader, show)

    else:

        init_dist('pytorch', **cfg.dist_params)

        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    return outputs

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def main():
    # Build dataset
    gpus = 1
    workers_per_gpu = 4
    nrows = 1e2
    chunksize = int(1e2)

    dataset = 'vit_files_img_xml_trunc'
    log_dataset = 'log_%s'%modele['conf']
    box_dataset = 'box_%s'%modele['conf']

    cfg = load_config(modele)

    img_df = load_data(dataset, nrows=nrows)
    print(img_df.shape)

    # load dataset on db and filter images never processed
    """
    img_db = read_dataframe(API_KEY_VIT, VERTICA_HOST, PROJECT_KEY_VIT, log_dataset, columns=['path','img_name'])
    img_db = img_db.eval('path + img_name').unique()
    print('there is %s images seen'%len(img_db))
    img_df = img_df.loc[~img_df.eval('path + img_name').isin(img_db),:]

    print("Need to process : %s images"%img_df.shape[0])
    """
    col_seg = ['img_name','path','x1','y1','x2','y2',
                                    'class','score']


    for chunk_df in chunker(img_df, chunksize):
        car_dataset = CustomDataset(chunk_df, ROOT_DIR,**cfg.data.val)
        print('length of dataset :')
        print(len(car_dataset))


        outputs, logs = run_detection(car_dataset, gpus ,workers_per_gpu, modele)

        img_seg = pd.DataFrame(columns=col_seg)
        log_seg = pd.DataFrame(columns=['img_name','path','traceback'])

        for (_,row), results, traceback  in zip(chunk_df.iterrows(), outputs, logs):
            to_save = save_result(results,
                        class_to_keep=class_to_keep,
                        dataset='coco',
                        score_thr=0.3)
            to_save_df = pd.DataFrame(to_save)
            to_save_df['img_name'] = row['img_name']
            to_save_df['path'] = row['path']
            print(to_save_df)
            img_seg = pd.concat([img_seg, to_save_df],
                    ignore_index=True,sort=False)

            log_seg = log_seg.append(dict(img_name=row['img_name'],
                                 path=row['path'],
                                 traceback=traceback),
                         ignore_index=True)

        img_seg['score'] = img_seg['score'].round(decimals=2)
        print('%s images processed'%log_seg.shape[0])
        write_dataframe(VERTICA_HOST,PROJECT_KEY_VIT,log_dataset,log_seg)
        write_dataframe(VERTICA_HOST,PROJECT_KEY_VIT,box_dataset,img_seg[col_seg])




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
