import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import torchvision
from torchvision import transforms
import pandas as pd
import gzip
import json
import os
from nltk.corpus import stopwords
import nltk
import heapq
from tqdm import tqdm
import numpy as np
import urllib
from io import BytesIO
import io
from PIL import Image
import torchvision.models as models
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pretrain resnext-101 model, remove last layer to extract features
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
model.eval()
model2 = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
model2.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path, category):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    data = pd.DataFrame.from_dict(df, orient='index')
    data = data[['title', 'image', 'asin']]
    data['category'] = category
    return data


data_dir = 'amazon/'

# items data
meta_app = getDF(os.path.join(data_dir, 'meta_Appliances.json.gz'), 'fashion')
meta_luxury = getDF(os.path.join(data_dir, 'meta_Luxury_Beauty.json.gz'), 'luxury')
meta_software = getDF(os.path.join(data_dir, 'meta_Software.json.gz'), 'software')

image_app = meta_app[["asin", "image", "category"]]
image_luxury = meta_luxury[["asin", "image", "category"]]
image_software = meta_software[["asin", "image", "category"]]

image_all = image_app.append(image_luxury, ignore_index=True)
image_all = image_all.append(image_software, ignore_index=True)

images = image_all.dropna(subset=['image'])


def extract_feature(img, model):
    # pytorch provides a function to convert PIL images to tensors.
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()

    # Read the image from file. Assuming it is in the same directory.
    pil_image = Image.open(img).convert('RGB')
    rgb_image = pil2tensor(pil_image)

    with torch.no_grad():
        feat = model(preprocess(rgb_image).unsqueeze(0).to(device))

    return feat

# import matplotlib.pyplot as plt
# # Plot the image here using matplotlib.
# def plot_image(tensor):
#     plt.figure()
#     # imshow needs a numpy array with the channel dimension
#     # as the the last dimension so we have to transpose things.
#     plt.imshow(tensor.numpy().transpose(1, 2, 0))
#     plt.show()

model_ft = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)

resnet_feature_list = []
items = []
images.columns
for i in tqdm(range(images.shape[0])):
    feature = None
    for j in range(len(images.iloc[i, 1])):
        img_path = images.iloc[i, 1][0]
        try:
            with urllib.request.urlopen(img_path) as url:
                f = io.BytesIO(url.read())
            feature = extract_feature(img_path, model)
            resnet_feature_list.append(feature.cpu().squeeze().detach().numpy())
            items.append(images.iloc[i, 0])
            break
        except:
            continue
    if feature is None:
        print('cannot find ', i)

resnet_feature_list = np.array(resnet_feature_list)
print(resnet_feature_list.shape)
# items = images['asin'].to_list()
res = {'items': items, 'img_features': resnet_feature_list}
with open('amazon/images.pickle', 'wb') as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
