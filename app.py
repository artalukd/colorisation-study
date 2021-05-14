

import io
import json
from io import StringIO
from io import BytesIO
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask import send_file
from flask import flash
# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
from PIL import Image
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader

from flask import redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)





class ColorizationNet(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 512

    ## First half: ResNet
    resnet = models.resnet34(num_classes=365) 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:8])

    ## Second half: Upsampling
    self.upsample1 = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.Upsample(scale_factor=2))
    self.upsample2 = nn.Sequential(
      nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2))
    
    self.upsample3 = nn.Sequential(
      nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2))
    self.upsample4 = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2)
      
      )
    self.output = nn.Sequential(
      nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
      nn.ReLU())
  def forward(self, input):

    result=[]
    for idx,model in enumerate(self.midlevel_resnet):
        input=model(input)
        if(idx in {2,4,5,6}):
            result.append(input)

    # Pass input through ResNet-gray to extract features
    upsample1 = self.upsample1(input)
    input = torch.cat((upsample1,result[-1]),1)
    upsample2 = self.upsample2(input)
    input = torch.cat((upsample2,result[-2]),1)
    upsample3 = self.upsample3(input)
    input = torch.cat((upsample3,result[-3]),1)
    upsample4 = self.upsample4(input)
    input = torch.cat((upsample4,result[-4]),1)
    output = self.output(input)

    # Upsample to get colors
    return output
model = ColorizationNet()
pretrained = torch.load('model_skip_best(1).pth', map_location=lambda storage, loc: storage)
model.load_state_dict(pretrained)
model.eval()

val_transforms = transforms.Compose([transforms.Resize((512,512))])
transforms_target = transforms.Compose([transforms.ToTensor(), transforms.Resize((512,512))])

def preprocess(image):
    img_original = val_transforms(image)
    img_original = np.asarray(img_original)
    img_lab = rgb2lab(img_original)
    img_lab = (img_lab + 128) / 255
    img_ab = img_lab[:, :, 1:3]
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
    img_original = rgb2gray(img_original)
    img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    return img_original,img_ab, (transforms_target(image))

def to_rgb(grayscale_input, ab_input, target,filename):

  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()
  plt.imsave(arr=grayscale_input, fname='static/uploads/gray_'+filename, cmap='gray')
  plt.imsave(arr=color_image, fname='static/uploads/color_'+filename)
  return (color_image, grayscale_input)


def get_prediction(image_bytes,filename):
    bw,ab,org = preprocess(Image.open(BytesIO(image_bytes)))
    dataset = TensorDataset(bw.unsqueeze(0), ab.unsqueeze(0), org.unsqueeze(0))
    loader = DataLoader( dataset, batch_size=1)

    for _, (input_gray, input_ab, target) in enumerate(loader):
        output_ab = model(input_gray) 
        return to_rgb(input_gray[0], output_ab[0].detach(), target[0].detach(),filename)

def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
    	file = request.files['file']
    	img_bytes = file.read()
    	filename = secure_filename(file.filename)
    	get_prediction(image_bytes=img_bytes,filename = filename)
    	return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        get_prediction(image_bytes=img_bytes)
        return send_file("color.jpg", mimetype='image/jpeg')
 
if __name__ == "__main__":
    app.run()
