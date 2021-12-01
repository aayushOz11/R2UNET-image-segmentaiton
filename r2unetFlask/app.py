from __future__ import division, print_function
import os
from flask.helpers import send_file
from keras.saving.model_config import model_from_json
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image as im
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #if u want to disable  GPU

ROWS= 320
COLS= 320
CHANNEL=3
m=1
# Define a flask app
app = Flask(__name__)
test_path ='C:\\Users\\Aayush Malde\\Desktop\\aayush documents\\TY KJSCE\\5th sem\\SC lab\\SC IA\\CHASE\\Test Images\\'
test_set_output='C:\\Users\\Aayush Malde\\Desktop\\aayush documents\\TY KJSCE\\5th sem\\SC lab\\SC IA\\CHASE\\Desired Output Test\\'
testing_set = [test_path+i for i in os.listdir(test_path)]
test_set_output_set = [test_set_output+i for i in os.listdir(test_set_output)]

def read_img(file_path):
    
    img = cv2.imread(file_path,cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS,COLS),interpolation=cv2.INTER_CUBIC)

def read_img1(file_path):
    
    img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (ROWS,COLS),interpolation=cv2.INTER_CUBIC)

def prepare_data(images,images1):
    m = len(images) #18
    X = np.zeros((m,ROWS,COLS,CHANNEL),dtype = np.float32) #(18,572,572)
    Y = np.zeros((m,ROWS,COLS),dtype = np.float32)
    
    for i,image_file in enumerate(images):
        img = read_img(image_file)
        X[i]= img
    for i,image_file in enumerate(images1):
        img = read_img1(image_file)
        Y[i]= img
    return X, Y
test_set_x, test_set_y = prepare_data(testing_set,test_set_output_set)

#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

def model_predict(img_path):
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    # use Keras model_from_json to make a loaded model

    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model

    loaded_model.load_weights("model_weights.h5")
    
    #taking user image and preprocessing on it
    X = np.zeros((m,ROWS,COLS,CHANNEL),dtype = np.float32)
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    print(img.shape)
    img=cv2.resize(img, (ROWS,COLS),interpolation=cv2.INTER_CUBIC)
    plt.imshow(img)
    plt.savefig('C:/Users/Aayush Malde/Desktop/aayush documents/djangoProj/retinaBloodVesselSegmentation/r2unetFlask/static/images/saved_figure2.jpeg')
    X[0]=img
    X = X.astype(np.float32)
    X=X/255

    print(img.shape)
    preds = loaded_model.predict(X,verbose=1)
    plt.imshow(np.squeeze(preds[0]))
    plt.savefig('C:/Users/Aayush Malde/Desktop/aayush documents/djangoProj/retinaBloodVesselSegmentation/r2unetFlask/static/images/svd_fig1.jpeg')
    pred_test_t=(preds>0.5).astype(np.int32)
    return pred_test_t


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds=[]
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        print(preds.shape)

        path1 = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
        print(path1)
        path1=path1+"\\r2unetFlask\\static\\images\\"
        plt.imshow(np.squeeze(preds[0]))
        # plt.show()
        plt.savefig('C:/Users/Aayush Malde/Desktop/aayush documents/djangoProj/retinaBloodVesselSegmentation/r2unetFlask/static/images/saved_figure1.jpeg')

        return ''

    return None 

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
