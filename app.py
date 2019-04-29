import os
from PIL import Image
import cv2
import json
from flask import Flask, render_template, Response, render_template_string, send_from_directory, request

import pandas as pd
from environment import ROOT_DIR
from filter import filter
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
        im = cv2.imread(filename)
        if x1 and x2 and y1 and y2:
            cv2.rectangle(im, (int(float(x1)), int(float(y1))), (int(float(x2)),int(float(y2))), (0,0,255), 2)
        if text:
            cv2.putText(im, text, (int(float(x1)), int(float(y2)) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if w and h:
            w, h = int(w), int(h)
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
    images = []
    dirname = os.path.dirname(csvpath)
    filename = os.path.join('/', csvpath)
    df_val = pd.read_csv(filename)
    filename = os.path.join('/', dirname, "predictions.csv")
    df_pred = pd.read_csv(filename)
    df = pd.concat([df_val, df_pred], axis=1)
    classes_ids = read_class_reference(dirname)
    df = df.assign(
            target_class=df['target'].apply(lambda x: classes_ids[str(int(x))]),
            pred_class=df['predictions'].apply(lambda x: classes_ids[str(int(x))])
    )
    df = df[df['target'] != df['predictions']]
    for i, row in df.iterrows():
        filename = os.path.join(ROOT_DIR,row['img_path'])
        im = Image.open(filename)
        w, h = im.size
        aspect = 1.0*w/h
        width = aspect * HEIGHT
        height = HEIGHT

        text = "Label:{} - Pred: {} Score: {:.3f}".format(row['target_class'], 
                row['pred_class'], row['score'])
        images.append({
            'width': int(width),
            'height': int(height),
            'src': filename,
            'x1': int(row["x1"]),
            'y1': int(row["y1"]),
            'x2': int(row["x2"]),
            'y2': int(row["y2"]),
            'text': text,
            })

    return render_template("preview.html", **{
        'images': images
    })


@app.route('/explore')
def images_explore():
    images = []
    df = filter(**request.args)
    print(df.head())
    for i, row in df.iterrows():
        filename = os.path.join(ROOT_DIR,row['path'],row['img_name'])
        im = Image.open(filename)
        w, h = im.size
        aspect = 1.0*w/h
        width = aspect * HEIGHT
        height = HEIGHT

        if ('marque' in row) and ('modele' in row):
            text = "{}, {}".format(row['marque'], row['modele'])
        else:
            text = 'Pas de prediction'

        images.append({
            'width': int(width),
            'height': int(height),
            'src': filename,
            'x1': row["x1"],
            'y1': row["y1"],
            'x2': row["x2"],
            'y2': row["y2"],
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
