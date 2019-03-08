#!/usr/bin/env python
import os
from PIL import Image
import cv2

from io import BytesIO
from flask import Flask, render_template, Response, render_template_string, send_from_directory, request

import pandas as pd
from environment import ROOT_DIR
from mmcv_patched import imshow_bboxes_custom
from filter import filter
WIDTH = 600
HEIGHT = 800

app = Flask(__name__)


@app.route('/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
        print(request.args)
    except (KeyError, ValueError):
        return send_from_directory('.', filename)

    try:
        #im = Image.open(filename)
        im = cv2.imread(filename)
        #im.thumbnail((w, h), Image.ANTIALIAS)
        im = cv2.resize(im,(w, h))#,interpolation=cv2.CV_INTER_AREA)
        #io = BytesIO()
        #im.save(io, format='JPEG')
        _, img_encoded = cv2.imencode('.jpg', im)
        #return Response(io.getvalue(), mimetype='image/jpeg')
        return Response(img_encoded.tobytes(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)


@app.route('/')
def images():
    images = []
    for root, dirs, files in os.walk('.'):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith('.jpg'):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0*w/h
            if aspect > 1.0*WIDTH/HEIGHT:
                width = min(w, WIDTH)
                height = width/aspect
            else:
                height = min(h, HEIGHT)
                width = height*aspect
            images.append({
                'width': int(width),
                'height': int(height),
                'src': filename
            })

    return render_template("preview.html", **{
        'images': images
    })


@app.route('/csv/<path:csvpath>')
def images_csv(csvpath):
    print(csvpath)
    images = []
    limit=1e2
    df = pd.read_csv(os.path.join('/',csvpath))
    df = df[df['target']==0]
    for i, row in df.iterrows():
        if i > limit:
            break
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
        images.append({
            'width': int(width),
            'height': int(height),
            'src': filename
        })
        i +=1
        print(row['img_path'])

    return render_template("preview.html", **{
        'images': images
    })


@app.route('/explore/<modele>/<status>')
def images_explore(modele,status):
    images = []
    limit=1e3
    df = filter(modele=[modele],status=[status],limit=limit,sampling=0.1)
    for i, row in df.iterrows():
        if i > limit:
            break
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
        images.append({
            'width': int(width),
            'height': int(height),
            'src': filename
        })
        i +=1
        print(filename)

    return render_template("preview.html", **{
        'images': images
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True,debug=True)
