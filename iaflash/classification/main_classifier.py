import argparse
import os
import random
import shutil
import time
import warnings
import sys

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

import iaflash
from iaflash.classification.custom_generator import DatasetDataframe, Crop
from iaflash.classification.simple_sampler import DistributedSimpleSampler
from iaflash.classification.utils import gather_evaluation

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to csv dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--val-csv', default=None, type=str,
                    help='csv to evaluate, default val.csv')

parser.add_argument('--root-dir', default=None, type=str,
                    help='root-dir, default ROOT_DIR')
best_acc1 = 0

# create model
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



# Wrapper of main(), called by python
def main_classifier(dict):
    class Args:
        def __getattr__(self, name):
            return None
    args = Args()
    for key, val in init_dict.items():
        setattr(args, key, val)
    for key, val in dict.items():
        setattr(args, key, val)
    return main(args)


def main(args):
    # paths definition
    if not args.root_dir :
        from iaflash.environment import ROOT_DIR
        args.root_dir = ROOT_DIR

    if not args.val_csv :
        args.val_csv  = os.path.join(args.data, 'val.csv')

    args.path_val_csv = os.path.dirname(args.val_csv)
    #args.name_val_csv =  os.path.splitext(os.path.basename(args.val_csv))[0]

    #


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print("SEE %s GPUs"%ngpus_per_node)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


    # gather all predictions
    predictions = gather_evaluation(os.path.join(args.path_val_csv ,'predictions.csv'), ngpus_per_node)
    probabilities = gather_evaluation(os.path.join(args.path_val_csv ,'probabilities.csv'), ngpus_per_node)
    targets = gather_evaluation(os.path.join(args.path_val_csv ,'targets.csv'), ngpus_per_node)
    indices = gather_evaluation(os.path.join(args.path_val_csv,'indices.csv'), ngpus_per_node)

    # confusion_matrix
    confusion = calculate_cm(probabilities.values, targets.values)
    print(confusion)

    # build results
    build_result(args.path_val_csv, args.val_csv)

def main_worker(gpu, ngpus_per_node, args):

    print("***" + str(args.workers) + "****")

    global best_acc1
    args.gpu = gpu

    # preload data
    filename = os.path.join(args.data, 'idx_to_class.json')
    with open(filename) as json_data:
        d = json.load(json_data)
        args.all_categories = [i for i in d.values()]

    #args.num_classes = dftrain['target'].unique().shape[0]
    args.num_classes = len(args.all_categories)
    print('%s Classes'%args.num_classes)


    valdir = os.path.join(args.val_csv)
    dfval = pd.read_csv(valdir,usecols=['img_path',  'x1', 'y1', 'x2', 'y2', 'score' ,'target'], index_col=False)
    #dfval = dfval[dfval['x1'].notnull()]
    assert (dfval['img_path'].notnull() & dfval['img_path']!='').any()

    dfval['target'] = dfval['target'].astype(int)
    print('%s Images in the val set'%dfval.shape[0])

    traindir = os.path.join(args.data, 'train.csv')
    dftrain = pd.read_csv(traindir,usecols=['img_path',  'x1', 'y1', 'x2', 'y2', 'score' ,'target'], index_col=False)
    assert (dftrain['img_path'].notnull() & dftrain['img_path']!='').any()
    dftrain['target'] = dftrain['target'].astype(int)
    print('%s Images in the train set'%dftrain.shape[0])


    class_weights = float(dftrain.shape[0]) / dftrain.groupby('target').size().sort_index()
    print(class_weights.shape)
    class_weights = torch.FloatTensor(class_weights.values)


    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        # This blocks until all processes have joined.
        print("world_size : %s"%args.world_size)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()


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


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(class_weights).cuda(args.gpu)

    cudnn.benchmark = True

    if args.gpu is not None:

        print("Use GPU: {} for training".format(args.gpu))

    # Data Loading Code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop = Crop()
    # iterate images
    train_dataset = DatasetDataframe(
        args.root_dir,
        dftrain,
        transforms.Compose([
            crop,
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = DatasetDataframe(
        args.root_dir,
        dfval,
        transforms.Compose([
            crop,
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = DistributedSimpleSampler(val_dataset)
        print('num_samples : %s' % val_sampler.num_samples)

    else:
        train_sampler = None
        val_sampler = None
    # shuffle to False not to have the data reshuffled at every epoch
    # wrap multiprocessing
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    '''
    val_loader = torch.utils.data.DataLoader(
        DatasetDataframe(args.root_dir, dfval, transforms.Compose([
            crop,
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # set epoch to deterministically shuffle data based on epoch
            # Every process will use its restricted data
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save only for one process
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            },  is_best, basename=args.data)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # watch the distribution
        # unique, counts = np.unique(target.cpu(non_blocking=True), return_counts=True)
        # print(dict(zip(unique, counts)))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # confusion matrix
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # every print_freq batch
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    confusion_matrix = torch.zeros(args.num_classes, args.num_classes)
    predictions = []
    probabilities = []
    targets = []
    # switch to evaluate mode
    model.eval()

    indices = list(val_loader.sampler)


    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # confusion matrix
            # confusion = calculate_cm_torch(output, target, args.num_classes)

            # accuracy
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            # confusion_matrix += confusion

            # save ppredictions
            _, preds = torch.max(output, 1)
            predictions.extend(preds.data.cpu().tolist())

            # save probabilities
            softmax = nn.Softmax()
            norm_output = softmax(output)
            probabilities.extend(norm_output.data.cpu().tolist())


            targets.extend(target.data.cpu().tolist())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))



        print("save predictions to %s"%os.path.join(args.path_val_csv))
        # Save predictions
        pd.DataFrame(predictions).to_csv(os.path.join(args.path_val_csv , 'predictions_%s.csv' % args.gpu), header=False, index=False)

        # Save probabilities
        pd.DataFrame(probabilities).to_csv(os.path.join(args.path_val_csv , 'probabilities_%s.csv' % args.gpu), header=False, index=False)

        # Save target
        pd.DataFrame(targets).to_csv(os.path.join(args.path_val_csv , 'targets_%s.csv' % args.gpu), header=False, index=False)

        # Save indices
        pd.DataFrame(indices).to_csv(os.path.join(args.path_val_csv , 'indices_%s.csv' % args.gpu), header=False, index=False)



        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, basename='/model', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(basename,filename))
    if is_best:
        shutil.copyfile(os.path.join(basename,filename), os.path.join(basename,'model_best.pth.tar') )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calculate_cm_torch(output, target, num_classes):
    """Calculate confussion matrix"""
    with torch.no_grad():
        confusion = torch.zeros(num_classes, num_classes)
        _, preds = torch.max(output, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

        return confusion

def calculate_cm(output, target):
    """Calculate confussion matrix"""
    confusion = np.zeros((output.shape[1], output.shape[1]))
    preds = np.argmax(output, axis=1)
    for t, p in zip(target.flatten(), preds.flatten()):
            confusion[int(t), int(p)] += 1

    return confusion

def create_simlink():

    import sys,random
    print(sys.path)

    from dss.api import read_dataframe

    from environment import ROOT_DIR, TMP_DIR, DSS_DIR, API_KEY_VIT,PROJECT_KEY_VIT, DSS_HOST, VERTICA_HOST

    dataset_name = 'CarteGrise_class'
    classes_list = ['RENAULT_CLIO', 'RENAULT_MEGANE', 'RENAULT_MEGANE SCENIC', 'PEUGEOT_308', 'CITROEN_C3', 'VOLKSWAGEN_GOLF', 'CITROEN_C4 PICASSO', 'PEUGEOT_207', 'PEUGEOT_208', 'PEUGEOT_3008']
    conditions = 'join_marque_modele IN (%s) ' %', '.join(["'%s'"%col for col in  classes_list])
    print(conditions)
    conditions += ' AND (DI_StatutDossier=4 OR DI_StatutDossier=6 OR DI_StatutDossier=13) '
    conditions += ' AND path IS NOT NULL '
    #conditions ='join_marque_modele IS NOT NULL AND (DI_StatutDossier=4 OR DI_StatutDossier=6 OR DI_StatutDossier=13) '
    columns = ['path','img1','join_marque_modele']
    limit = 1e7
    sampling = -0.1
    #DSS_HOST = VERTICA_HOST+":0"
    df = read_dataframe(API_KEY_VIT,VERTICA_HOST,PROJECT_KEY_VIT,dataset_name,columns,conditions,limit,sampling)

    df = df[df.img1.notnull()]

    print('There is %s images'%df.shape[0])
    CARS_DIR = '/data/cars'
    shutil.rmtree(CARS_DIR)
    os.makedirs(CARS_DIR,exist_ok=True)


    # create class
    df_class = df.groupby('join_marque_modele')

    for modele, df_class in df_class:

        for i, row in df_class.iterrows():
            if random.uniform(0, 1)>0.3:
                set = 'train'
            else:
                set = 'val'
            #print(row['path'],1.996row['img1'])
            src = os.path.join(args.root_dir,row['path'],row['img1'])
            base_dst = os.path.join(CARS_DIR,set,modele)
            if not os.path.exists(base_dst):
                os.makedirs(base_dst,exist_ok=True)
            dst = os.path.join(base_dst,row['img1'])
            if not os.path.exists(dst):
                os.symlink(src, dst)

    return

if __name__ == '__main__':
    #create_simlink()
    args = parser.parse_args()
    main(args)

"""
python main_classifier.py -a resnet18 --lr 0.01 --batch-size 256  --pretrained --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

python main_classifier.py -a resnet18 --lr 0.01 --batch-size 256  --pretrained --evaluate --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/cars
python main_classifier.py -a resnet18 --lr 0.01 --batch-size 256  --pretrained --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0  /model/resnet18-100-2
python main_classifier.py -a resnet18 --lr 0.01 --batch-size 256  --pretrained --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0   /model/resnet18-102
python main_classifier.py -a resnet18 --lr 0.01 --batch-size 256  --pretrained --evaluate --resume /model/model_best.pth.tar --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0   /model/resnet18-102
"""
