import pandas as pd
import os
from iaflash.environment import ROOT_DIR, TMP_DIR, DSS_DIR, API_KEY_VIT, PROJECT_KEY_VIT, DSS_HOST, VERTICA_HOST
from mmdet.core import get_classes
import numpy as np

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



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

def load_data(dataset_name = 'img_MIF',nrows=1e3):
    # cached the dataset as csv
    dataset_path = os.path.join(DSS_DIR,'{}.csv'.format(dataset_name))
    if not os.path.isfile(dataset_path):
        img_MIF_df = read_dataframe(API_KEY_VIT,VERTICA_HOST,PROJECT_KEY_VIT,dataset_name, columns=['path','img1','img2'])
        img_MIF_df.to_csv(dataset_path)
    else:
        print('Read cached csv : %s'%dataset_path)
        img_MIF_df = pd.read_csv(dataset_path, nrows=nrows)


    print(img_MIF_df.head())
    print('%s rows have been retrieved'%img_MIF_df.shape[0])

    if 'img1' in img_MIF_df.columns and 'img2' in img_MIF_df.columns:
        img_MIF_df = img_MIF_df.assign(
        img1_path=(ROOT_DIR + img_MIF_df['path'] + "/" + img_MIF_df['img1']),
        img2_path=(ROOT_DIR + img_MIF_df['path'] + "/" + img_MIF_df['img2']),
                        )

        img_df = pd.melt(img_MIF_df,
            id_vars='path',
            value_vars=['img1','img2'], # list of days of the week
            var_name='img_num',
            value_name='img_name').sort_values('path')

    # filter only .jpg extension

    img_df = img_df[(img_df.img_name != "") & (img_df.path != "")]
    img_df.dropna(subset=['path','img_name'],inplace=True,how='any')
    img_df = img_df[img_df.img_name.str.contains('.jpg')]

    img_df.reset_index(inplace=True)

    return img_df
