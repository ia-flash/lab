
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from .custom_generator import DatasetDataframe, Crop
from .simple_sampler import DistributedSimpleSampler



def gather_evaluation(filename, gpu):
    base_filename, file_extension = os.path.splitext(filename)
    df = pd.DataFrame()
    print(gpu)

    if os.path.isfile(filename):
        return pd.read_csv(filename, header=None)

    for i in range(gpu) :
        filename_i = base_filename + '_%s'%i + file_extension
        print('read %s'%filename_i)
        df_i = pd.read_csv(filename_i,header=None)
        df = pd.concat([df, df_i], ignore_index=True,axis=0)
        os.remove(filename_i)

    df.to_csv(filename, header=False, index=False)
    return df



def build_result(model_path):
    """
    """
    predictions = pd.read_csv(os.path.join(model_path, 'predictions.csv'),header=None)
    probabilities =  pd.read_csv(os.path.join(model_path, 'probabilities.csv'),header=None)
    targets =  pd.read_csv(os.path.join(model_path, 'targets.csv'),header=None)
    indices =  pd.read_csv(os.path.join(model_path, 'indices.csv'),header=None)

    results = pd.concat([indices.rename(columns={0:'index'}),
           predictions.rename(columns={0:'pred_class'}),
           probabilities.max(axis=1).rename('proba')], axis=1)

    results.drop_duplicates(subset = 'index', inplace=True) # remove added rows to complete batch
    results.set_index('index', inplace=True)

    """
    # TODO : Add dataset to Args
    dataset = 'train.csv'
    df = pd.read_csv(model_path+'/dataset',usecols=['img_path',  'x1', 'y1', 'x2', 'y2', 'score' ,'target'], index_col=False)
    results = pd.merge(results, df, how='inner', left_index=True, right_index=True)
    """
    results.to_csv(model_path+'/results.csv', index=False)

    return results

def refine()    :
    """ Refine train.csv without error of tagging
    """
    return

def calculate_cm(output, target):
    """Calculate confussion matrix"""
    confusion = np.zeros((output.shape[1], output.shape[1]))
    preds = np.argmax(output, axis=1)
    for t, p in zip(target.flatten(), preds.flatten()):
            confusion[int(t), int(p)] += 1

    return confusion


def dict2args(dict):
    class Args:
        def __getattr__(self, name):
            return None

    args = Args()

    for key, val in dict.items():
        setattr(args, key, val)

    return args

    
def load_model(args):

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()


    # freeze
    #for param in model.parameters():
    #    param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default

    model.fc = nn.Linear(512, args.num_classes) # assuming that the fc7 layer has 512 neurons, otherwise change it

    if args.distributed:
        print("Distributed")
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # by default if evaluate and no resume file, take best_model.pth.tar in args.data
    if (not args.resume) and args.evaluate :
        print('Take model_best.pth.tar in : %s'%args.data)
        args.resume = os.path.join(args.data, "model_best.pth.tar")

    # optionally resume from a checkpoint
    if args.resume :
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    return model, args

def data_loader():
    return

def evaluate(model, val_loader):
    return
