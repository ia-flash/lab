import argparse
import os.path as osp
import shutil, time, os
import tempfile
import pandas as pd

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import wrap_fp16_model
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


from iaflash.detection.utils import load_data, chunker, save_result
from iaflash.environment import ROOT_DIR, TMP_DIR, DSS_DIR, API_KEY_VIT, PROJECT_KEY_VIT, DSS_HOST, VERTICA_HOST
from iaflash.detection.generator_detection import CustomDataset
from iaflash.dss.api import read_dataframe, write_dataframe

print(os.environ['CUDA_HOME'])
print(os.environ['CUDA_VISIBLE_DEVICES'])

class_to_keep = ['person','bicycle', 'car',
                'motorcycle','bus',
                'truck','traffic_light','stop_sign',
                'parking_meter','bench']

col_seg = ['img_name','path','x1','y1','x2','y2',
                                    'class','score']

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results, logs= [], []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        traceback = ''
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


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results, logs= [], []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    for i, data in enumerate(data_loader):
        traceback = ''
        try :
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
        except Exception as traceback:
            print(traceback)
            result = []

        logs.append(traceback)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    return results, logs


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    modele= osp.splitext(osp.basename(args.config))[0]
    print(modele)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        print('custom import')
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        print("not distributed")

    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        print("distributed")

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    nrows = 100
    chunksize = int(4*1e3)
    dataset = 'vit_files_img_xml_trunc'
    log_dataset = 'log_%s_test'%modele
    box_dataset = 'box_%s_test'%modele
    img_df = load_data(dataset, nrows=nrows)
    print(img_df.shape)

    # Reprise sur incident
    #img_db = read_dataframe(API_KEY_VIT, VERTICA_HOST, PROJECT_KEY_VIT, log_dataset, columns=['path','img_name'])
    #img_db = img_db.eval('path + img_name').unique()
    #print('there is %s images seen'%len(img_db))
    #img_df = img_df.loc[~img_df.eval('path + img_name').isin(img_db),:]

    print("Need to process : %s images"%img_df.shape[0])


    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        #outputs = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda(),
                        device_ids=[torch.cuda.current_device()],
                        broadcast_buffers=False)
        #outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    for chunk_df in chunker(img_df, chunksize):

        img_seg = pd.DataFrame(columns=col_seg)
        log_seg = pd.DataFrame(columns=['img_name','path','traceback'])

        car_dataset = CustomDataset(chunk_df, ROOT_DIR,**cfg.data.val)
        print('length of dataset :')
        print(len(car_dataset))

        data_loader = build_dataloader(
            car_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        if not distributed:
            outputs, logs = single_gpu_test(model, data_loader, args.show)
        else:
            outputs, logs = multi_gpu_test(model, data_loader)

        indices = list(data_loader.sampler)

        # Convert and save
        for (_,row), results, traceback  in zip(chunk_df.iloc[indices,:].iterrows(), outputs, logs):
            to_save = save_result(results,
                        class_to_keep=class_to_keep,
                        dataset='coco',
                        score_thr=0.3)
            to_save_df = pd.DataFrame(to_save)
            to_save_df['img_name'] = row['img_name']
            to_save_df['path'] = row['path']

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
