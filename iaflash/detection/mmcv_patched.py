import cv2
import numpy as np
import pandas as pd

import mmcv
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val

from matplotlib import pyplot as plt

def imshow(img, win_name='', wait_time=0,ax=None):
    """Show an image.
    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    # cv2.imwrite("/data/test/test_annoted.jpg",img)
    img2 = img[:,:,::-1]
    #print(img2.shape)
    if ax is not None:
        ax.imshow(img2)
        ax.axis('off')
    """
    else:
        plt.imshow(img2)
    """


def imshow_bboxes_custom(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        label_text = _bboxes[:,4]
        color_text = color_val('green')
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)
            cv2.putText(img, '{:.02f}'.format(label_text[j]), (_bboxes[j, 0], _bboxes[j, 3] - 2), cv2.FONT_HERSHEY_COMPLEX, 1, color_text)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=5,
                      font_scale=2,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None,
                      ax=None):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[3] - 3),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        imshow(img, win_name, wait_time,ax)
    if out_file is not None:
        imwrite(img, out_file)

from mmdet.core import get_classes
def show_result(img, result, dataset='coco', score_thr=0.3,ax=None,out_file=None):
    class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(img)
    imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,ax=ax,
        out_file=out_file)


if __name__ == '__main__':
    df = pd.read_csv('./dss/bbox_small.csv')
    df['count'] = df.groupby('img_name').cumcount()+1
    path = "/data/sources/verbalisations/antai/img/"
    for _, grp in df.groupby('img_name'):
        img = imread("{}{}/{}".format(path, grp['path'].iloc[0], grp['img_name'].iloc[0]))
        cols = ['x1', 'y1', 'x2', 'y2', 'score']
        bboxes = grp[(grp['class'] == 'car') & (grp['score'] > 0.5)][cols].values
        print(bboxes)
        out_file = "/data/sources/verbalisations/antai/crop_img/{}".format(grp['img_name'].iloc[0])
        if (len(bboxes) > 0):
            imshow_bboxes_custom(img, bboxes, out_file=out_file)
    #for _, val in df.iterrows():
    #    if val['class'] == 'car':
    #        print(val)
    #        path = "/data/sources/verbalisations/antai/img/"
    #        img = imread(path + val['path'] + '/' + val['img_name'])
    #        bboxes = np.array([[val['x1'], val['y1'], val['x2'], val['y2'], val['score']]])
    #        img_name = val['img_name'].split(".")[0] + str(val["count"]) + ".jpg"
    #        out_file = "/data/sources/verbalisations/antai/crop_img/{}".format(img_name)
    #        imshow_bboxes(img, bboxes, out_file=out_file)
