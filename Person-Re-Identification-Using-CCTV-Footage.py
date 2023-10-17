#!/usr/bin/env python
# coding: utf-8

# <h1><center>Person Re-Identification Using CCTV Footage</h1>

# <h2>computer vision with PyTorch : Siamese Network</h2>

# <h2>Siamese Network

# ![Sample Image](1_bJABur9wzFNACosQkim8kw.webp)
# 

# <h3>Download and Import libraries

# In[2]:


get_ipython().system('pip instaLWP:Sll segmentation-models-pytorch -q')
get_ipython().system('pip install -U git+https://github.com/albumentations-team/albumentations -q')
get_ipython().system('pip install --upgrade opencv-contrib-python -q')


# In[3]:


get_ipython().system('git clone https://github.com/parth1620/Person-Re-Id-Dataset')


# In[4]:


import sys
sys.path.append("/kaggle/working/Person-Re-Id-Dataset")


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

"""
Timm: PyTorch Image Models (timm) is a library for state-of-the-art-image classification, containing a collection of image models, optimizers, schedulers, augmentations and much more.
"""
import timm

import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from skimage import io
from sklearn.model_selection import train_test_split

"""
tqdm is a library that is used for creating Python Progress Bars. It gets its name from the Arabic name taqaddum, which means 'progress. '
"""
from tqdm import tqdm


# <h3>Configurations

# In[7]:


DATA_DIR = "/kaggle/input/market-1501/Market-1501-v15.09.15/bounding_box_train/"
CSV_FILE = "/kaggle/working/Person-Re-Id-Dataset/train.csv"

BATCH_SIZE = 32
LR = 0.001
EPOCHS = 15

DEVICE = 'cuda'


# In[8]:


df = pd.read_csv(CSV_FILE)
df.head()


# In[9]:


row = df.iloc[11]



A_img = io.imread(DATA_DIR + row.Anchor)
P_img = io.imread(DATA_DIR + row.Positive)
N_img = io.imread(DATA_DIR + row.Negative)


# In[10]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10,5))

ax1.set_title("Anchor")
ax1.imshow(A_img)

ax2.set_title("Positive")
ax2.imshow(P_img)

ax3.set_title("Negative")
ax3.imshow(N_img)


# In[11]:


train_df, valid_df = train_test_split(df, test_size = 0.20, random_state = 42)


# <h3>Create APN Dataset

# In[12]:


class APN_Dataset(Dataset):
    
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        
        A_img = io.imread(DATA_DIR + row.Anchor)
        P_img = io.imread(DATA_DIR + row.Positive)
        N_img = io.imread(DATA_DIR + row.Negative)
        
        A_img = torch.from_numpy(A_img).permute(2, 0 ,1) / 255.0
        P_img = torch.from_numpy(P_img).permute(2, 0 ,1) / 255.0
        N_img = torch.from_numpy(N_img).permute(2, 0 ,1) / 255.0
        
        return A_img, P_img, N_img


# In[13]:


trainset = APN_Dataset(train_df)
validset = APN_Dataset(valid_df)

print(f"Size of trainset : {len(trainset)}")
print(f"Size of validset : {len(validset)}")


# In[14]:


idx = 40
A,P,N = trainset[idx]

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize= (10,5))

ax1.set_title('Anchor')
ax1.imshow(A.numpy().transpose((1,2,0)), cmap = 'gray')

ax2.set_title('Positive')
ax2.imshow(P.numpy().transpose((1,2,0)), cmap = 'gray')

ax3.set_title('Negative')
ax3.imshow(N.numpy().transpose((1,2,0)), cmap = 'gray')


# <h3>Load Dataset into Batches

# In[15]:


trainloader = DataLoader(trainset, batch_size = BATCH_SIZE,shuffle = True)
validloader = DataLoader(validset, batch_size = BATCH_SIZE)


# In[16]:


print(f"No. of batches in trainloader : {len(trainloader)}")
print(f"No. of batches in validloader : {len(validloader)}")


# In[17]:


for A, P, N in trainloader:
    break;

print(f"One image batch shape : {A.shape}")


# <h3>Create Model

# In[18]:


class APN_Model(nn.Module):
    
    def __init__(self, emb_size = 512):
        super(APN_Model, self).__init__()
        
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(in_features = self.efficientnet.classifier.in_features,
                                                out_features = emb_size)
        
    
    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings


# In[19]:


model = APN_Model()
model.to(DEVICE)


# <h3>Create Train and Eval Function

# In[20]:


def train_fn(model, dataloader, optimizer, criterion):
    model.train() # ON Dropout
    total_loss = 0.0 
    
    for A,P,N in tqdm(dataloader):
        A,P,N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)
        
        A_embs = model(A)
        P_embs = model(P)
        N_embs = model(N)
        
        loss = criterion(A_embs, P_embs, N_embs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


# In[21]:


def eval_fn(model, dataloader, criterion):
    model.eval()  # OFF Dropout
    total_loss = 0.0 
    
    with torch.no_grad():
        for A,P,N in tqdm(dataloader):
            A,P,N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)

            A_embs = model(A)
            P_embs = model(P)
            N_embs = model(N)

            loss = criterion(A_embs, P_embs, N_embs)

            total_loss += loss.item()
        
        return total_loss / len(dataloader)


# In[22]:


criterion = nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)


# <h3>Create Training Loop

# In[23]:


best_valid_loss = np.Inf

for i in range(EPOCHS):
    train_loss = train_fn(model, trainloader, optimizer, criterion)
    valid_loss = eval_fn(model, validloader, criterion)
    
    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), "best_model.pt")
        best_valid_loss = valid_loss
        print("SAVED_WEIGHT_SUCCESS")
    
    print(f"EPOCHS: {i+1} train_loss: {train_loss} valid_loss: {valid_loss}")


# <h3>Get Anchor Embeddings

# In[24]:


def get_encoding_csv(model, anc_img_names):
    anc_img_names_arr = np.array(anc_img_names)
    encodings = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(anc_img_names_arr):
            A = io.imread(DATA_DIR + i)
            A = torch.from_numpy(A).permute(2, 0, 1) / 255.0
            A = A. to(DEVICE)
            A_enc = model(A.unsqueeze(0)) # c,h,w --> (1,c,h,w)
            encodings.append(A_enc.squeeze().cpu().detach().numpy())
        
        encodings = np.array(encodings)
        encodings = pd.DataFrame(encodings)
        df_enc = pd.concat([anc_img_names, encodings], axis=1)
    
    return df_enc


# In[25]:


model.load_state_dict(torch.load("best_model.pt"))
df_enc = get_encoding_csv(model, df["Anchor"])


# In[26]:


df_enc.to_csv("database.csv", index=False)
df_enc.head()


# <h3>Inference

# In[27]:


def euclidean_dist(img_enc, anc_enc_arr):
    dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc - anc_enc_arr).T))
    return dist


# In[28]:


idx = 0
img_name = df_enc["Anchor"].iloc[idx]
img_path = DATA_DIR + img_name

img = io.imread(img_path)
img = torch.from_numpy(img).permute(2, 0, 1) / 255.0

model.eval()
with torch.no_grad():
    img = img.to(DEVICE)
    img_enc = model(img.unsqueeze(0))
    img_enc = img_enc.detach().cpu().numpy()


# In[29]:


anc_enc_arr = df_enc.iloc[:, 1:].to_numpy()
anc_img_names = df_enc["Anchor"]


# In[30]:


distance = []

for i in range(anc_enc_arr.shape[0]):
    dist = euclidean_dist(img_enc, anc_enc_arr[i : i+1, :])
    distance = np.append(distance, dist)


# In[31]:


closest_idx = np.argsort(distance)


# In[32]:


from utils import plot_closest_imgs

plot_closest_imgs(anc_img_names, DATA_DIR, img, img_path, closest_idx, distance, no_of_closest = 10);


# In[ ]:




