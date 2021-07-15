from flask import Flask, render_template, Response, request, render_template,Flask, session
import cv2
import numpy as np
import pandas as pd
import os 
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from IPython.display import Image
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, redirect, url_for
import json
import time
from PIL import Image
import sys
from model_training.siam_cosian import SiameseNetwork
from multiprocessing import Value
from flask_session import Session


#app configuration
app = Flask(__name__)
video = cv2.VideoCapture(0)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

#image transformation for neural network
transform = transforms.Compose([                                
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((48,48)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5081), (0.2552))])

#function for video streaming from webcam
def gen(video):
    while True:
        success, image = video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#helping function for video streaming from webcam
def get_image(video):
    while True:
        success, image = video.read()
        
        return image
        
  
filename=os.path.join(app.root_path, 'static', 'img')
UPLOAD_FOLDER = filename
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



counter = Value('i', 0)
results=[]
#message={}
#main route
@app.route("/", methods=['GET', 'POST'])
def index():
    
    device = torch.device('cpu')
    model=SiameseNetwork()
    model.load_state_dict(torch.load('cosian.pt', map_location=device))
    model.eval()    
    
    if request.method == 'POST':
        
        print("Step number "+str(counter.value))
        print('Incoming..')
        print(request.get_json())  
        
        path=request.get_json()
        print(path)
        global video
        img=get_image(video)
        
        #for extracting face from the webcam image
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")    
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(48, 48)
        )
        print("Found {0} Faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            
            img = gray[y:y + h, x:x + w]
            print("[INFO] Object found.")
            break

        
        img=cv2.resize(img, (48, 48))        
        img2 = cv2.imread(path['src'], 0)
        img2=cv2.resize(img2, (48, 48))
        

        img = Image.fromarray(img)
        img2 = Image.fromarray(img2)
        img=transform(img)
        img=img.unsqueeze(0)
        img2=transform(img2)
        img2=img2.unsqueeze(0)
        output1,output2 = model(img,img2)
        pdist = nn.PairwiseDistance()
        pred = pdist(output1, output2)
        print(pred)
        prob = torch.sigmoid(pred)
        print(prob)        
        results.append(prob)
        
        
        if counter.value ==7:
            val=sum(results)/len(results)
            val=val.detach().numpy()[0]
            print("here")            
            print(val)            
            print(str(val))             
            p=str(val)            
            session['result'] = p
            session.modified = True
            print(session['result'])
            return redirect(url_for('result'))
        counter.value += 1
        #print("==-=-=-=-=-==-=-=-=-==")
        #print(results)
        #print("==-=-=-=-=-==-=-=-=-==")
    
        
    
    return render_template('index.html')


#route when result is shown
@app.route('/result')
def result():    
    res=session['result']
    return render_template('result.html',val=res)

def cv_get_im(src):
    print(src)
    img2 = cv2.imread(src, 0)
    return img2


def hello():    
    message={}
    if request.method == 'POST':
        print('Incoming..')
        print(request.get_json())  # parse as JSON          
        message=request.get_json()        
        return message

#video streaming route
@app.route('/video_feed')
def video_feed():
    global video
   
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':    
    app.run(host='0.0.0.0', port=2204, threaded=True,debug=True)
