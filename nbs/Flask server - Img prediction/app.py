"""
Run the Flask SERVER:

    FLASK_ENV=development FLASK_APP=app.py flask run

CLIENT (command line):

    curl -X POST -F my_img_file=@cardigan.jpg http://localhost:5000/predict
    
CLIENT (Python):

    import requests
    resp = requests.post("http://localhost:5000/predict",
                     files={"my_img_file": open('cardigan.jpg','rb')})
Example response:

    {
      "cardigan": 0.7083, 
      "wool": 0.0837, 
      "suit": 0.0431, 
      "Windsor_tie": 0.031, 
      "trench_coat": 0.0307
    }

"""

import numpy as np
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import json


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

classes = json.load(open('imagenet_classes.json'))
model = models.densenet121(pretrained=True)
model.eval()

def pre_process(image_file):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.open(image_file)
    return my_transforms(image).unsqueeze(0) # unsqueeze is for the BS dim

def post_process(logits):
    vals, idxs = logits.softmax(1).topk(5)
    vals = vals[0].numpy()
    idxs = idxs[0].numpy()
    result = {}
    for idx, val in zip(idxs, vals):
        result[classes[idx]] = round(float(val), 4)
    return result

def get_prediction(image_file):
    with torch.no_grad():
        image_tensor  = pre_process(image_file)
        output = model.forward(image_tensor)
        return post_process(output)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['my_img_file']
        result_dict = get_prediction(image_file)
        #return jsonify(result_dict)
        #return json.dumps(result_dict)
        return result_dict

if __name__ == '__main__':
    app.run()