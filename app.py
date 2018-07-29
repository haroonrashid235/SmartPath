import csv
import scipy.misc
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image


from flask import Flask, render_template, url_for, request, redirect, send_file, send_from_directory, flash
from werkzeug.utils import secure_filename

from pager import Pager
import ScriptOne

def read_table(url):
    """Return a list of dict"""
    # r = requests.get(url)
    with open(url) as f:
        return [row for row in csv.DictReader(f.readlines())]


APPNAME = "SmartPathology"
STATIC_FOLDER = 'examples'
TABLE_FILE = "examples/catalog.csv"
save_path = 'predicitons/'
UPLOAD_FOLDER = 'examples/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

table = read_table(TABLE_FILE)
pager = Pager(len(table))

filename = ""


app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config.update(
APPNAME=APPNAME,
)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return redirect('/0')

@app.route('/<int:ind>/')
def image_view(ind=None):
    table = read_table(TABLE_FILE)
    pager = Pager(len(table))
    if ind >= pager.count:
        return render_template("404.html"), 404
    else:
        pager.current = ind
        return render_template(
            'imageview.html',
            index=ind,
            pager=pager,
            data=table[ind])

index = 14
@app.route('/goto', methods=['POST', 'GET'])    
def goto():
    return redirect('/' + request.form['index'])

@app.route('/im', methods=['GET'])
def img():
    return send_file('outfile.jpg')    

@app.route('/predict', methods=['POST', 'GET'])    
def predict():
    image_path = request.form['predict_btn']
    prediction = ScriptOne.prediction('.' + image_path)
    output_file = 'outfile.jpg'
    scipy.misc.imsave(output_file, prediction)
    #return redirect('/im',messages={'image_path' : image_path})
    return redirect('/im')
    #return render_template('predict.html', path_original = image_path, path_predict= '/im')

@app.route('/after_upload', methods=['POST','GET'])
def after_upload():
    global index
    file = request.files['file']
    filename = secure_filename(file.filename)
    k = filename.rfind('.')
    filename = filename[:k] + '.jpg'
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    data = np.asarray(img, dtype="int32" )
    height, width, depth = data.shape
    index += 1
    name = filename[:k]
    new_row = '\n' + str(index) + ',' + name + ',' + str(height) + ',' + str(width)+ ',' + str(depth)
    fd = open('examples/catalog.csv','a')
    fd.write(new_row)
    fd.close()
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return render_template('upload.html')

if __name__ == '__main__':
    app.secret_key = 'secret_smart_path'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
