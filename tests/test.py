import os
import pandas as pd
from iaflash.filter import filter
from iaflash.classification.main_classifier import main_classifier
from iaflash.classification.utils import build_result

def test_filter_cmd():
    cmd =  """python  ../iaflash/filter.py \
    --table CarteGrise_norm_melt_joined2 \
    --status 4 6 13\
    --dir /model/test \
    --nb_classes 10 \
    --score 0.95 \
    --sampling 0.001 \
    --not-null path img_name x1 \
    --limit 0  \
    --where '(TYPEEQUIP_Libelle='ETC' AND img_name LIKE '%_1.jpg') OR (TYPEEQUIP_Libelle!='ETC')'"""

    os.system(cmd)
    df_train = pd.read_csv('/model/test/train.csv')
    assert df_train.shape[0]>10

def test_filter(sampling=0.001, nb_classes=10):
    not_null = ['path', 'img_name', 'x1']
    filter(table='CarteGrise_norm_melt_joined2', status=[4, 6, 13], dir='/model/test',
        sampling=sampling, nb_classes=nb_classes, score=0.95, not_null=not_null,
        limit=0,
        where = "(TYPEEQUIP_Libelle='ETC' AND img_name LIKE '%_1.jpg') OR (TYPEEQUIP_Libelle!='ETC')")

    df_train = pd.read_csv('/model/test/train.csv')
    df_val = pd.read_csv('/model/test/val.csv')

    assert df_train.shape[0]>10
    assert not df_train[['img_path','x1']].isnull().any().any()
    assert df_val.shape[0]>10

def test_classification_cmd():
    cmd =  """python ../iaflash/classification/main_classifier.py \
    -a resnet18 --lr 0.01 --batch-size 128  --pretrained --epochs 15\
    --resume /model/test/model_best.pth.tar\
    --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
    /model/test"""

    # TRAIN
    os.system(cmd)

    # VERIFY
    test_path = '/model/test_square'

    results = build_result(test_path, test_path + '/val.csv')
    acc1 = (results['pred_class'] == results['target'] ).sum() / float(results.shape[0])
    print(acc1)
    assert acc1 > 0.4, "Accuracy is too low"


if __name__ == '__main__':
    #test_filter(sampling=0.005, nb_classes=20)
    test_filter()
    #test_classification_cmd()
