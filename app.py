#!/usr/bin/env python
import os
from PIL import Image
import cv2
import json

from io import BytesIO
from flask import Flask, render_template, Response, render_template_string, send_from_directory, request

import pandas as pd
from environment import ROOT_DIR
from mmcv_patched import imshow_bboxes_custom
from filter import filter
WIDTH = 600
HEIGHT = 400

app = Flask(__name__)


@app.route('/<path:filename>')
def image(filename):
    try:
        meta = request.args['meta']
        meta = json.loads(meta)
        print(meta)
        w = int(request.args['width'])
        h = int(request.args['height'])
        x1 = int(meta['x1'])
        y1 = int(meta['y1'])
        x2 = int(meta['x2'])
        y2 = int(meta['y2'])
    except (KeyError, ValueError):
        print('cannot access to keys')
        return send_from_directory('.', filename)

    try:
        #im = Image.open(filename)
        im = cv2.imread(filename)
        #im.thumbnail((w, h), Image.ANTIALIAS)
        cv2.rectangle(im, (x1, y1), (x2,y2), (0,0,255), 2)
        #cv2.putText(im, text, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #    1, (0,255,0), 2)
        im = cv2.resize(im,(w, h)) #,interpolation=cv2.CV_INTER_AREA)
        _, img_encoded = cv2.imencode('.jpg', im)
        #return Response(io.getvalue(), mimetype='image/jpeg')
        return Response(img_encoded.tobytes(), mimetype='image/jpeg')

    except IOError:
        abort(404)

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
        if aspect > 1.0*WIDTH/HEIGHT:
            width = min(w, WIDTH)
            height = width/aspect
        else:
            height = min(h, HEIGHT)
            width = height*aspect
        #text = "Label:{} - Pred: {} Score: {:.3f}".format(row['target_class'], row['pred_class'], row['score'])
        dict_img = {
            'width': int(width),
            'height': int(height),
            'src': filename,
            #'text': text,
            }
        dict_img.update({'meta':row.to_dict()})
        images.append(dict_img)

    return render_template("preview.html", **{
        'images': images
    })


@app.route('/explore')
def images_explore():
    images = []
    df = filter(**request.args)
    for i, row in df.iterrows():
        filename = os.path.join(ROOT_DIR,row['path'],row['img_name'])
        im = Image.open(filename)
        w, h = im.size
        aspect = 1.0*w/h
        if aspect > 1.0*WIDTH/HEIGHT:
            width = min(w, WIDTH)
            height = width/aspect
        else:
            height = min(h, HEIGHT)
            width = height*aspect
        text = "{}, {}".format(row['marque'].lower(), row['modele'])
        images.append({
            'width': int(width),
            'height': int(height),
            'src': filename

        }.update(row.to_dict()))

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
