import argparse
import os
import random
import shutil
import time
import warnings
import sys
import json
import pandas as pd

from iaflash.environment import ROOT_DIR, TMP_DIR, DSS_DIR, API_KEY_VIT,PROJECT_KEY_VIT, DSS_HOST, VERTICA_HOST
from iaflash.dss.api import read_dataframe


dataset_name = 'bbox_marque_class_class'
columns = ['path','img_name','x1','y1','x2','y2','score','modele','marque','class']
limit = 1e5
sampling = 0.1
radar_type = {
        'ETE' : 'equipement terrain embarqu',
        'ETF' : 'equipement terrain fixe',
        'ETC' : 'equipement terrain chantier',
        'ETVM' : 'equipement terrain vitesse moyenne',
        'ETM' : 'equipement terrain mobile',
        'ETVLPL' : 'equipement terrain discriminant pl vl'
        }

not_null = ['path', 'img_name']

parser = argparse.ArgumentParser(description='Filter Dataset From DSS and Write it')

parser.add_argument('--dir', metavar='DIR',
                    help='path to csv dataset')

parser.add_argument('--keep',dest='keep',action='store_true',
                    help='Drop and Create')

parser.add_argument( '--table', metavar='TABLE',type=str, default=dataset_name,
                    help='Number of lines to keep')

parser.add_argument('--status', metavar='STATUS DOSSIER',type=int, nargs='+',
                    choices=range(60),
                    help='Filter DI_StatutDossier')

parser.add_argument( '--score', metavar='SCORE',type=float,
                    help='Filter by fuzzy match score')

parser.add_argument( '--nb_classes', metavar='NOMBRE CLASSES',type=int,
                    help='Filter Nombre de classes')

parser.add_argument( '--class_list', metavar='LIST CLASSES',type=str,
                    help='Classes to use')


parser.add_argument( '--modele', metavar='modele',type=str, nargs='+',
                    help='Filter modele')

parser.add_argument('--sens', metavar='SENS DETECTION',type=str, nargs='+',
                    choices=['ELOI','RAPP','BI-DIRECTIONNEL'],
                    help='Filter modele')

parser.add_argument('--radar', metavar='RADAR TYPE',type=str, nargs='+',
                    choices=radar_type.keys(),
                    help='Filter Modele')

parser.add_argument('--columns', metavar='COLUMNS',type=str, nargs='+',
                    help='Columns to retrieve',default=columns)

parser.add_argument('--not-null', type=str, nargs='+',
                    help='Filter those columns if null',default=not_null)

parser.add_argument('--where', metavar='WHERE',type=str,
                    help='WHERE CLAUSE',default=None)

parser.add_argument('--sampling', metavar='SAMPLE',type=float,default=sampling,
                    help='Sampling rate when extracting')

parser.add_argument('-l', '--limit', metavar='LIMIT',type=int, default=limit,
                    help='Number of lines to keep')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument( '--shuffle', dest='shuffle',action='store_true',
                    help='shuffle all rows after request the data')

parser.add_argument( '--api-key',
                    help='DSS api key', default=API_KEY_VIT)

parser.add_argument( '--vertica-host',
                    help='VERTICA Host', default=VERTICA_HOST)

parser.add_argument( '--project-key',
                    help='DSS project key', default=PROJECT_KEY_VIT)

parser.add_argument( '--connector',
                    help='Connector choose beteewn vertica or postgres', default='vertica')

print('Read args')

def main(args):

    if (not args.keep) or (not os.path.exists(args.dir)):
        print("Drop and recreate everything in %s ?" %args.dir )
        print('[Y/n]')
        choice = input().lower()
        if choice == "n" :
            print('=> Escape')
            return (args.dir)
        else :
            print('Erasing...')
            shutil.rmtree(args.dir, ignore_errors=True)
            os.makedirs(args.dir, exist_ok=True)

    df = read_df(args)
    print('Retrieve %s rows'%df.shape[0])
    df = create_mapping(args, df)
    write_df(args, df)

    return(args.dir)


def dict2args(dict):

    args = parser.parse_args()

    for key, val in vars(args).items():
        print(key, val)
        if key not in dict.keys():
            atr = val
        else:
            atr = dict[key]

        setattr(args, key, atr)
        print("set attribute %s for key %s "%(val, atr))

    return args

def filter(**filt_dict):
    # init Namespace object from parser init states

    dict2args(filt_dict)
    main(args)
    """
    return read_df(args)
    """
def read_df(args):
    conditions = ''

    if args.columns :
        if type(args.columns) is str:
            args.columns = ["\"%s\""%col for col in args.columns.split(",")]


    if args.class_list:
        print(args.class_list)
        #class_list = pd.read_csv(os.path.join(args.dir, args.modele_list))
        class_list = pd.read_csv(args.class_list, header=None)
        conditions += 'class IN ({}) '.format(', '.join(["'{}'".format(i) for i in class_list.values.flatten()]))

    print('ok')
    if args.score:
        if type(args.score) is str:
            args.score = args.score.split(",")
        if conditions != '':
            conditions += ' AND '
        conditions += 'score_sivnorm > {} '.format(args.score)

    if args.modele :
        if type(args.modele) is str:
            args.modele = args.modele.split(",")
        if conditions != '':
            conditions += ' AND '
        conditions += 'class IN ({}) '.format(', '.join(["'%s'"%col for col in  args.modele]))

    if args.status :
        if type(args.status) is str:
            args.status = args.status.split(",")
        if conditions != '':
            conditions += ' AND '
        conditions += '"DI_StatutDossier" IN ({}) '.format(', '.join([str(int(col)) for col in  args.status]))

    if args.sens :
        if type(args.sens) is str:
            args.sens = args.sens.split(",")
        if conditions != '':
            conditions += ' AND '
        conditions += '"csa.Params_Leg.Sens_Detect" IN (%s) ' %', '.join(["'%s'"%col for col in  args.sens])

    if args.radar :
        if type(args.radar) is str:
            args.radar = args.radar.split(",")
        conditions += ' AND '
        conditions += 'TYPEEQUIP_Libelle IN (%s) ' %', '.join(["'%s'"%radar_type[col] for col in  args.radar])

    if args.where :
        if conditions != '':
            conditions += ' AND '
        conditions +=  '(' + args.where + ')'


    if conditions != '':
        conditions += ' AND '

    if args.not_null:
        conditions_not_null = ' AND '.join(['%s IS NOT NULL'%col for col in args.not_null])
        print(conditions_not_null)
        conditions += conditions_not_null


    if args.nb_classes:
        if type(args.nb_classes) is str:
            args.nb_classes = args.nb_classes.split(",")
        group_req = conditions + ' AND ' + "class ILIKE class GROUP BY class  ORDER BY COUNT(class) DESC LIMIT {};".format(args.nb_classes)
        df = read_dataframe(API_KEY_VIT,VERTICA_HOST,PROJECT_KEY_VIT,
            args.table,['class, COUNT(class)'],group_req, args.connector)
        df['class'].to_csv(os.path.join(args.dir, 'classes.csv'), index=False)
        conditions += ' AND '
        conditions += 'class IN ({}) '.format(', '.join(["'{}'".format(i) for i in df['class'].tolist()]))

    #conditions ='join_marque_modele IS NOT NULL AND (DI_StatutDossier=4 OR DI_StatutDossier=6 OR DI_StatutDossier=13) '
    #DSS_HOST = VERTICA_HOST+":1000    print('There is %s images'%df.shape[0])
    df = read_dataframe(args.api_key,args.vertica_host,args.project_key,
        args.table,args.columns,conditions,args.limit,args.sampling, args.connector)

    df = df[df.notnull()]
    print("Drop duplicates ...")
    df.drop_duplicates('img_name',inplace=True)

    if args.shuffle : # shuffle but take care that img1 and img2 are bounded
        print('Start shufffing data ...')
        start = time.time()
        groups = [df for _, df in df.groupby('path')]
        random.shuffle(groups)
        df = pd.concat(groups).reset_index(drop=True)
        end = time.time()
        print("Took %s"%(end-start))

    return df

def create_mapping(args,df):
    #df_class = df.groupby('_rank',sort=True)
    df_class = df.groupby('class')

    i = 0
    idx_to_class = {}

    for _rank, tmp in df_class:
        assert tmp['class'].unique().shape[0] == 1
        class_name = tmp['class'].unique()[0]
        df.loc[tmp.index,'target'] = int(i)
        idx_to_class.update({i:class_name})
        i+=1
        print('Count %s images for the class: %s'%(len(tmp.index),class_name))

    with open(os.path.join(args.dir,'idx_to_class.json'), 'w') as outfile:
        json.dump(idx_to_class,outfile)

    print("Number classes : %s"%str(i))
    return df

def write_df(args,df):

    df['img_path'] = df['path'] +'/' + df['img_name']
    #df.rename(columns={'_rank':'target'},inplace=True)
    cols = ['img_path','target','x1','y1','x2','y2','score']

    if args.evaluate:
        df[cols].to_csv(os.path.join(args.dir,'val.csv'), index=False)
    else:
        nb_rows = len(df)
        df.loc[0:int(nb_rows*0.7),cols].to_csv(os.path.join(args.dir,'train.csv'), index=False)
        df.loc[int(nb_rows*0.7):int(nb_rows*0.9),cols].to_csv(os.path.join(args.dir,'val.csv'), index=False)
        df.loc[int(nb_rows*0.9):nb_rows,cols].to_csv(os.path.join(args.dir,'test.csv'), index=False)
        print('save csv to %s'%args.dir)
        #df[cols].tail(int(df.shape[0]*(1/3.))).to_csv(os.path.join(args.dir,'val.csv'), index=False)
        #df[cols].head(int(df.shape[0]*(2/3.))).to_csv(os.path.join(args.dir,'train.csv'), index=False)

    # create mapping


def test_filter():

    # Choose yout filter
    filt_dict = dict(modele=['CLIO','MEGANE'],
        sampling=0.1,
        radar=['ETF'],
        sens=['ELOI'],
        columns=columns+['TYPEEQUIP_Libelle'],
        limit=10)


    df = filter(**filt_dict)
    assert df.shape == (filt_dict['limit'],len(filt_dict['columns'])), '{} not match with query'.format(df)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

"""
python filter.py --table CarteGrise_norm_melt_joined --status 4,6,13 -l 10 --dir /model/test/
python filter.py --sampling 0.1  --class CLIO 206 --radar ETF --sens ELOI RAPP -l 10 /model/test/
python filter.py --status 0  --sens ELOI RAPP -l 10 --dir /model/test/
python filter.py --status 0  --sens ELOI RAPP -l 10 --dir /model/test/ --evaluate

python filter.py --table CarteGrise_norm_melt_joined --status 4 6 13 --dir /model/resnet18-101 --class_list classes.csv --keep --sampling 0 --limit 0 --score 0.95
python filter.py --table CarteGrise_norm_melt_joined --status 4 6 13 --dir /model/test --nb_classes 20 --score 0.95 --sampling 0.001

# resnet18-150
python filter.py --table CarteGrise_norm_melt_joined2 --status 4 6 13 --dir /model/resnet18-150 --nb_classes 160 --score 0.95  --sampling 0 --limit 0   --where "(TYPEEQUIP_Libelle='ETC' AND img_name LIKE '%_1.jpg') OR (TYPEEQUIP_Libelle!='ETC')"

# resnet18-151
python -m iaflash.filter --table cartegrise_norm_melt_joined_psql --status 4 6 13 --dir /model/resnet18-151  \
--class_list /model/classes-2018-151.csv  --sampling 0 --limit 0 --score 0.95 \
--where "(\"TYPEEQUIP_Libelle\"='ETC' AND img_name LIKE '%_1.jpg') OR (\"TYPEEQUIP_Libelle\"!='ETC')" \
--api-key dummyKey_VBfL5V9czGa9a77kcVzmwZcQvdjt8oBk \
--project-key 'VIT3' \
--vertica-host 192.168.4.25 \
--connector postgres
--shuffle
--not-null x1 path img_name


"""
