import os
from PIL import Image
import cv2
import json
from flask import Flask, render_template, Response, render_template_string, send_from_directory, request

import pandas as pd
from iaflash.environment import ROOT_DIR
from iaflash.filter import read_df, dict2args
WIDTH = 600
HEIGHT = 400

app = Flask(__name__)


@app.route('/<path:filename>')
def image(filename):
    w = request.args.get('w', None)
    h = request.args.get('h', None)
    x1 = request.args.get('x1', None)
    y1 = request.args.get('y1', None)
    x2 = request.args.get('x2', None)
    y2 = request.args.get('y2', None)
    marque = request.args.get('marque', None)
    modele = request.args.get('modele', None)
    score = request.args.get('score', None)

    text = request.args.get('text', None)

    try:
        print("filename is : " + filename)
        im = cv2.imread(os.path.join('/',filename))
        if x1 and x2 and y1 and y2:
            cv2.rectangle(im, (int(float(x1)), int(float(y1))), (int(float(x2)),int(float(y2))), (0,0,255), 2)
        if text:
            cv2.putText(im, text, (int(float(x1)), int(float(y2)) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if w and h:
            w, h = int(w), int(h)
            print(im.shape)
            im = cv2.resize(im, (w, h))
        _, img_encoded = cv2.imencode('.jpg', im)
        return Response(img_encoded.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        print(e)

    return send_from_directory('.', filename)


@app.route('/')
def images():
    return json.dumps({'status': 'ok'})


@app.route('/csv/<path:csvpath>')
def images_csv(csvpath):
    query = request.args.get('query', None)
    limit = request.args.get('limit', None)
    if limit:
        nrows = int(limit)
    else:
        nrows = None

    images = []
    dirname = os.path.dirname(csvpath)
    filename = os.path.join('/', csvpath)
    df = pd.read_csv(filename, nrows=nrows)#.sample(10000)
    #df_val = df_val[df_val['x1'].notnull()]

    classes_ids = read_class_reference(dirname)
    df['class_name'] = df['target'].astype(int).astype(str).replace(classes_ids)
    df['text'] = 'Label: ' + df['class_name']
    if'pred_class' in df.columns :
        df['text'] += '- Pred: ' +  df['pred_class'].astype(int).astype(str).replace(classes_ids)
    if 'proba' in df.columns :
        df['text'] +=  ' Score: ' + df['proba'].round(3).astype(str)
    #df = df[df['target'] != df['pred_class']]
    if query:
        print(query)
        df = df.query(query)

    """
    if limit :
        df = df.sample(int(limit))
    """

    df['img_path'] = df['img_path'].astype(str)
    df = df.sort_values(by =['img_path'],  ascending=False)

    print(df.head())
    for i, row in df.iterrows():
        ROOT_DIR = "/vgdata/sources/verbalisations/antai"
        filename = os.path.join(ROOT_DIR, row['img_path'])
        im = Image.open(filename)
        w, h = im.size
        aspect = 1.0*w/h
        width = aspect * HEIGHT
        height = HEIGHT

        images.append({
            'width': int(width),
            'height': int(height),
            'src': filename,
            'x1': row.get("x1",0),
            'y1': row.get("y1",0),
            'x2': row.get("x2",0),
            'y2': row.get("y2",0),
            'text': row['text'],
            })

    return render_template("preview.html", **{
        'images': images
    })


@app.route('/explore')
def images_explore():
    ROOT_DIR = '/vgdata/sources/verbalisations/antai'
    images = []
    df = read_df(dict2args(request.args))
    print(df.head())
    col_img = request.args.get('col_img', 'img_name')
    query = request.args.get('query', None)

    if query:
        print(query)
        df = df.query(query)

    df.sort_values('path',inplace=True)


    for col in ['x1','y1','x2','y2'] :
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for i, row in df.iterrows():
        filename = os.path.join(ROOT_DIR,row['path'],row[col_img])
        im = Image.open(filename)
        w, h = im.size
        aspect = 1.0*w/h
        width = aspect * HEIGHT
        height = HEIGHT

        #if ('marque' in row) and ('modele' in row):
        #    text = "{}, {}".format(row['marque'], row['modele'])
        if ('CG_MarqueVehicule' in row) and ('CG_ModeleVehicule' in row):
            text = "{}, {}".format(row['CG_MarqueVehicule'], row['CG_ModeleVehicule'])
            row = row.append(pd.Series([30,30,30,30], index=['x1','y1','x2','y2']))
        elif ('class' in row) and ('score' in row):
            text = "{}, {}".format(row['class'], row['score'])
        else:
            text = 'Pas de prediction'

        images.append({
            'width': int(width),
            'height': int(height),
            'src': filename,
            'x1': row.get("x1",0),
            'y1': row.get("y1",0),
            'x2': row.get("x2",0),
            'y2': row.get("y2",0),
            'text': text
        })

    return render_template("preview.html", **{
        'images': images
    })

def read_class_reference(dirname):
    filename = os.path.join('/', dirname, 'idx_to_class.json')
    with open(filename) as json_data:
        classes_ids = json.load(json_data)
    return classes_ids

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True,debug=True)
