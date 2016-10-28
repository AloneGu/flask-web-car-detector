import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug import secure_filename
from config import *
from car_detection import CarDetector


global curr_file_name
curr_file_name = None
global car_worker
car_worker = CarDetector()


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RES_FOLDER):
    os.makedirs(RES_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RES_FOLDER'] = RES_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global curr_file_name
    global car_worker
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            curr_file_name = filename
            curr_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(curr_file_path)
            car_worker.run('image',curr_file_path) # process this image
            return redirect(url_for('upload_file'))
    return render_template('index.html',fn=curr_file_name)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
@app.route('/results/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['RES_FOLDER'],filename)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5001)